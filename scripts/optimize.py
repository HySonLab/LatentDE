import argparse
import itertools
import importlib
import os
import rootutils
import torch
from copy import deepcopy
from lightning import seed_everything
from torch import Tensor
from tqdm import tqdm
from typing import List

rootutils.setup_root(os.path.dirname(__file__),
                     indicator=".project-root",
                     pythonpath=True)

from src.common.utils import (   # noqa: E402
    edit_distance,
    parse_module_name_from_path,
    print_stats
)
from src.models import CNNVAE, GruVAE, BaseVAE   # noqa: E402
from src.models.oracles import (      # noqa: E402
    ESM2_Landscape, ESM2_Attention, ESM1b_Landscape
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize sequences.")
    parser.add_argument("config_file", type=str, help="Path to config module")
    parser.add_argument("--ref_file", type=str, help="Path to reference sequence.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--model_ckpt_path", type=str, help="Checkpoint of model.")
    parser.add_argument("--oracle_ckpt_path", type=str, help="Checkpoint of oracle.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--steps", type=int, default=500, help="# gradient ascent steps.")
    parser.add_argument("--num_samples", type=int, default=20, help="# optimized sequences.")
    parser.add_argument("--beam", type=int, default=4, help="Beam size.")
    parser.add_argument("--num_batch", type=int, default=1, help="Number of batches.")
    parser.add_argument("--num_gen", type=int, default=200, help="# directed evolution generation.")
    parser.add_argument("--scale", type=float, default=1.0, help="Noise for random exploration.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=os.path.abspath("../exps/results_no_active"),
                        help="Output directory.")
    parser.add_argument("--prefix", type=str, default="", help="output prefix.")
    parser.add_argument("--eval", action="store_true", help="Evaluate final population with ESM1b.")
    args = parser.parse_args()
    return args


def grad_ascent_latent(module, latents: List[Tensor] | Tensor, lr: float) -> List[Tensor]:

    def main_process(latent: Tensor):
        # Define optimize
        optimizer = torch.optim.Adam([latent], lr=lr)

        pbar = tqdm(range(args.steps))
        for i in pbar:
            optimizer.zero_grad()
            fitness = module.predict_property_from_latent(latent)
            if i % 100 == 0:
                score = fitness.detach().cpu().mean().squeeze().tolist()
                pbar.set_postfix({"fitness": score})
            fitness = (-fitness).mean()
            fitness.backward()
            optimizer.step()

        return latent

    if isinstance(latents, Tensor):
        return main_process(latents)
    else:
        opt_latents = [main_process(latent) for latent in latents]
        return opt_latents


def initialize_models(dec_type: str,
                      model_ckpt: str,
                      oracle_ckpt: str,
                      device: torch.device):
    # main model
    if dec_type == "rnn":
        vae = GruVAE
    elif dec_type == "cnn":
        vae = CNNVAE
    else:
        raise ValueError(f"{dec_type} is not supported.")
    module = vae.load_from_checkpoint(model_ckpt, map_location=device, device=device)
    module.eval()

    # fitness oracle
    pred_decoder = ESM2_Attention(hidden_dim=1280)  # fixed for now
    predictor = ESM2_Landscape.load_from_checkpoint(
        oracle_ckpt, map_location=device, net=pred_decoder
    )
    predictor.eval()

    return module, predictor


def initialize_eval_oracle(task: str, device: torch.device):
    return ESM1b_Landscape(task, device)


def perform_directed_evolution(
    module: BaseVAE, oracle: ESM2_Landscape, seqs: List[str], num_iter: int, wt_seq: str,
    batch: int, scale: float, beam_size: int, keep_size: int, device: torch.device
):
    init_scores = oracle.infer_fitness_glob(seqs, device, batch)
    items = list(zip(seqs, init_scores))
    items = sorted(items, key=lambda x: x[1], reverse=True)[:keep_size]
    cur_items = items

    factor = 0.1
    pbar = tqdm(range(num_iter))
    pbar.set_postfix({"max_score": cur_items[0][1],
                      "dist": edit_distance(cur_items[0][0], wt_seq)})
    for i in pbar:
        # multiply sequences by beam size
        cur_items = list(itertools.chain.from_iterable(
            list(deepcopy(it) for _ in range(beam_size))
            for it in cur_items
        ))
        cur_seqs = list(map(lambda x: x[0], cur_items))

        # perform directed evolution by "reconstructing" sequences
        new_seqs = module.reconstruct_from_wt_glob(cur_seqs, scale, factor, i, batch)
        new_scores = oracle.infer_fitness_glob(new_seqs, device, batch)
        new_items = list(zip(new_seqs, new_scores))

        # sort and filter out sequences with low scores
        new_items = cur_items + new_items
        if i == num_iter - 1:
            keep_size = keep_size * beam_size
        cur_items = sorted(new_items, key=lambda x: x[1], reverse=True)[:keep_size]

        # log to cmd
        pbar.set_postfix({"max_score": cur_items[0][1],
                          "dist": edit_distance(cur_items[0][0], wt_seq)})

    final_seqs, final_scores = list(zip(*cur_items))

    return final_seqs, final_scores


def main(args):
    # Create cfg
    cfg = importlib.import_module(parse_module_name_from_path(args.config_file))
    # general config
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
    device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")

    # Config to save output
    task = os.path.basename(args.ref_file).split("_")[0]
    output_dir = os.path.join(args.output_dir, task)
    filename = f"{args.prefix}_{cfg.decoder_type}_{'freeze' if cfg.freeze_encoder else ''}_" \
               f"{'relso' if cfg.model_kwargs['use_neg_sampling'] else 'reg'}_" \
               f"lr={args.lr}_scale={args.scale}_{args.seed}"

    os.makedirs(output_dir, exist_ok=True)
    with open(args.ref_file, "r") as f:
        wt_seq = f.readlines()[0]

    assert args.num_samples % args.num_batch == 0
    batch_size = args.num_samples // args.num_batch
    results = []
    # =================== #
    # ====== Model ====== #
    # =================== #
    module, predictor = initialize_models(cfg.decoder_type,
                                          args.model_ckpt_path,
                                          args.oracle_ckpt_path,
                                          device)
    oracle = initialize_eval_oracle(task, device) if args.eval else None

    # =================================== #
    # ====== Optimize latent space ====== #
    # =================================== #
    # Generate latent
    seqs = [wt_seq for _ in range(args.num_samples)]
    if args.num_samples == 1:
        latents, *_ = module.encode(seqs)
        latents = latents.detach()
        latents.requires_grad = True
    else:
        latents = []
        for i in range(0, len(seqs), batch_size):
            seq = seqs[i:i + batch_size]
            latent, *_ = module.encode(seq)
            latent = latent.detach()
            latent.requires_grad = True
            latents.append(latent)

    # Optimize latent
    print("**********\nGradient Ascent only:")
    latents = grad_ascent_latent(module, latents, args.lr)

    # Produce optimized sequence thru gradient ascent.
    opt_seqs = module.generate_from_latent(latents)

    fitness = predictor.infer_fitness_glob(opt_seqs, device, batch_size)

    eval_fitness = oracle.infer_fitness(opt_seqs, device) if args.eval else None
    stats = print_stats(opt_seqs, fitness, wt_seq, eval_fitness)
    results.extend(["**********\nGA only:", stats, "\n"])

    # ======================================== #
    # ====== Perform directed evolution ====== #
    # ======================================== #
    print("**********\nGradient Ascent + Directed Evolution:")
    opt_seqs, fitness = perform_directed_evolution(module, predictor, opt_seqs, args.num_gen,
                                                   wt_seq, batch_size, args.scale, args.beam,
                                                   args.num_samples, device)
    eval_fitness = oracle.infer_fitness(opt_seqs, device) if args.eval else None
    stats = print_stats(opt_seqs, fitness, wt_seq, eval_fitness)
    results.extend(["**********\nGA + DE:", stats, "\n"])

    # ================================================== #
    # ====== Perform directed evolution (DE Only) ====== #
    # ================================================== #
    print("**********\nDirected Evolution only:")
    opt_seqs = [wt_seq] * args.num_samples
    opt_seqs, fitness = perform_directed_evolution(module, predictor, opt_seqs, args.num_gen,
                                                   wt_seq, batch_size, args.scale, args.beam,
                                                   args.num_samples, device)

    eval_fitness = oracle.infer_fitness(opt_seqs, device) if args.eval else None
    stats = print_stats(opt_seqs, fitness, wt_seq, eval_fitness)
    results.extend(["**********\nGA + DE:", stats, "\n"])

    with open(f"{output_dir}/{filename}.txt", "w") as f:
        f.write("\n".join(results))

    print("Experiment Completed")


if __name__ == "__main__":
    args = parse_args()
    main(args)
