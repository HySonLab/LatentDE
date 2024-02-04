import argparse
import importlib
import os
import rootutils
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lightning import seed_everything
from sklearn.manifold import TSNE

import matplotlib.tri as mtri

rootutils.setup_root(os.path.dirname(__file__),
                     indicator=".project-root",
                     pythonpath=True)

from src.common.utils import parse_module_name_from_path  # noqa: E402
from src.models import CNNVAE, GruVAE  # noqa: E402
from src.dataio.proteins import ProteinsDataModule  # noqa: E402
from src.models.oracles import (  # noqa: E402
    ESM2_Landscape, ESM2_Attention, ESM1b_Landscape
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model.")
    parser.add_argument("config_file", type=str, help="Path to config module")
    parser.add_argument("--csv_file", type=str, help="Path to CSV data.")
    parser.add_argument("--ref_file",
                        type=str,
                        help="Path to reference sequence.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--model_ckpt_path",
                        type=str,
                        help="Checkpoint of model.")
    parser.add_argument("--oracle_ckpt_path",
                        type=str,
                        help="Checkpoint of oracle.")
    parser.add_argument("--save_dir", type=str, help="Save path.")
    parser.add_argument("--is_predicted", type=int, help="0 or 1")
    parser.add_argument("--split", type=str, help="train or val")
    args = parser.parse_args()
    return args


def parse_dict_from_module(module):
    start = False
    module_dict = module.__dict__
    newdict = {}
    for key in module_dict.keys():
        if start:
            newdict[key] = module_dict[key]
        if key != "os":
            continue
        else:
            start = True

    return newdict


def initialize_models(dec_type: str, model_ckpt: str, oracle_ckpt: str,
                      task: str, device: torch.device):
    # main model
    if dec_type == "rnn":
        vae = GruVAE
    elif dec_type == "cnn":
        vae = CNNVAE
    else:
        raise ValueError(f"{dec_type} is not supported.")
    module = vae.load_from_checkpoint(model_ckpt,
                                      map_location=device,
                                      device=device)
    module.eval()

    # fitness oracle
    if oracle_ckpt is not None:
        pred_decoder = ESM2_Attention(hidden_dim=1280)
        predictor = ESM2_Landscape.load_from_checkpoint(oracle_ckpt,
                                                        map_location=device,
                                                        net=pred_decoder)
        predictor.eval()
    else:
        predictor = ESM1b_Landscape(task, device)

    return module, predictor


def plot_latent_space(latent_codes, labels, dataname):
    print("Reducing dimension to 2")
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes = tsne.fit_transform(latent_codes)
    print("TSNE done")
    plt.scatter(latent_codes[:, 0],
                latent_codes[:, 1],
                c=labels,
                cmap='viridis')
    plt.colorbar()
    plt.title(f'Protein Landscape of {dataname}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(f"{args.save_dir}/{dataname}.pdf")


def plot_3D_latent_space(latent_codes, labels, dataname):
    print("Reducing dimension to 2")
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes = tsne.fit_transform(latent_codes)
    print("TSNE done")
    X, Y = np.meshgrid(latent_codes[:, 0], latent_codes[:, 1])
    Z = np.array(labels)
    y = Z.reshape(-1, 1)
    h = Z * y
    print(Z.shape)

    plt.contourf(latent_codes[:, 0], latent_codes[:, 1], h, cmap='viridis')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(latent_codes[:, 0], latent_codes[:, 1], latent_codes[:, 2], c=labels, cmap='viridis')
    # plt.colorbar()
    plt.title(f'Protein Landscape of {dataname}')
    # plt.set_xlabel('t-SNE Dimension 1')
    # plt.set_ylabel('t-SNE Dimension 2')
    # plt.set_zlabel("t-SNE Dimension 3")
    plt.savefig(f"{args.save_dir}/3D_{dataname}.png")


def plot_landscape(latent_codes, labels, dataname, is_predicted):
    print("Reducing dimension to 2")
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes = tsne.fit_transform(latent_codes)
    print("TSNE done")
    x = latent_codes[:, 0]
    y = latent_codes[:, 1]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    triang = mtri.Triangulation(x, y)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_trisurf(triang, labels, cmap=plt.cm.CMRmap)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel("fitness")
    suffix = "sim" if is_predicted else "gt"
    plt.savefig(
        f"{args.save_dir}/landscape_{dataname}_{suffix}.png",
        bbox_inches="tight",
        transparent=True
    )


def main(args):
    # Create cfg
    cfg = importlib.import_module(parse_module_name_from_path(
        args.config_file))
    # general config
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
    accelerator = "cpu" if args.devices == "-1" else "gpu"
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")

    # ================== #
    # ====== Data ====== #
    # ================== #
    data_kwargs = cfg.data_kwargs
    data_kwargs.update({
        "csv_data": args.csv_file,
        "seed": args.seed,
        "train_batch_size": args.batch_size,
        "valid_batch_size": args.batch_size,
    })
    datamodule = ProteinsDataModule(**data_kwargs)

    max_length = datamodule.max_length
    if cfg.decoder_type == "rnn":
        upsampler_low_res_dim = \
            int(max_length / cfg.upsampler_stride**cfg.upsampler_num_deconv_layers) + 1
        max_len = upsampler_low_res_dim * cfg.upsampler_stride**cfg.upsampler_num_deconv_layers
        assert max_len >= max_length, "new max_len must be higher than old one."
        max_length = max_len
        datamodule.set_max_length(max_length)

    task = os.path.basename(args.ref_file).split("_")[0]
    module, predictor = initialize_models(cfg.decoder_type,
                                          args.model_ckpt_path,
                                          args.oracle_ckpt_path, task, device)

    datamodule.setup(None)

    if args.split == "train":
        print("Use training set")
        loader = datamodule.train_dataloader()
    elif args.split == 'val':
        print("Use validation set")
        loader = datamodule.val_dataloader()

    all_zs = []
    all_ys = []
    for batch in tqdm(loader, total=len(loader)):
        seqs, fitness = batch['sequences'], batch['fitness']
        with torch.no_grad():
            z, _, _ = module.encode(seqs)
            if args.is_predicted:
                fitness = module.predict_property_from_latent(z).view(-1)
        all_zs.append(z.cpu().numpy())
        all_ys.append(fitness.cpu().numpy())

    latent_codes = np.concatenate(all_zs, axis=0)
    labels = np.concatenate(all_ys, axis=0)
    print(latent_codes.shape)
    print(labels.shape)
    plot_landscape(latent_codes, labels, task, args.is_predicted)


if __name__ == "__main__":
    args = parse_args()
    main(args)
