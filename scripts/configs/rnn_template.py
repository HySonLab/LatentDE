decoder_type = "rnn"

"""
======================
===== VAE CONFIG =====
======================
"""
# Encoder
encoder_kwargs = {
    "pretrained_encoder_path": "facebook/esm2_t30_150M_UR50D"
}
# Latent
latent_kwargs = {
    "latent_dim": 320
}
# Upsampler
upsampler_kwargs = {
    "upsampler_max_filter": 512,
    "upsampler_kernel_size": 2,
    "upsampler_min_deconv_dim": 32,
    "upsampler_num_deconv_layers": 3,
    "upsampler_stride": 2,
    "upsampler_padding": 0,
    "upsampler_dropout": 0.2,
    "upsampler_act_type": "relu",
}
upsampler_stride = upsampler_kwargs["upsampler_stride"]
upsampler_num_deconv_layers = upsampler_kwargs["upsampler_num_deconv_layers"]
# Decoder
decoder_kwargs = {
    "dec_num_layers": 1,
    "dec_hidden_dim": 512,
    "gru_dropout": 0.0,
    "inp_dropout": 0.4,
    "dec_attn_method": "general",    # ["dot", "concat", "general"]
    "use_teacher_forcing": 0.5
}
# Predictor
predictor_kwargs = {
    "pred_hidden_dim": 512,
    "pred_dropout": 0.2,
}
# Others
model_kwargs = {
    "max_len": None,
    "nll_weight": 1.0,
    "mse_weight": 1.0,
    "kl_weight": 1.0,
    "beta_min": 0.0,
    "beta_max": 1.0,
    "Kp": 0.01,   # 0.01
    "Ki": 0.0001,
    "lr": 0.0002,
    "reduction": "mean",
}

freeze_encoder = True


"""
======================
===== DATAMODULE =====
======================
"""
data_kwargs = {
    "train_val_split": (0.9, 0.1),
    "num_workers": 64,
}


"""
=================
===== OTHER =====
=================
"""
output_dir = "./exps"
wandb_project = "LatentDE"
num_ckpts = 3
save_every_n_epochs = 1
precision = "highest"
