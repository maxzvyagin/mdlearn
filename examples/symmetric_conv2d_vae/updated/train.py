import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from mdlearn.utils import parse_args, BaseSettings
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
import glob
import pdb


class SymmetricConv2dVAEConfig(BaseSettings):
    # File paths
    # Path to HDF5 training file
    input_path: Path = Path("TODO")
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("TODO")
    # Optionally resume training from a checkpoint file
    checkpoint_path: Optional[Path] = None

    input_shape: Tuple[int, int, int] = (1, 926, 926)
    filters: List[int] = [64, 64, 64, 64]
    kernels: List[int] = [5, 3, 3, 3]
    strides: List[int] = [2, 2, 2, 2]
    affine_widths: List[int] = [128]
    affine_dropouts: List[float] = [0.0]
    latent_dim: int = 10
    activation: str = "ReLU"
    output_activation: str = "Sigmoid"
    lambda_rec: float = 1.0
    seed: int = 42
    in_gpu_memory: bool = False
    num_data_workers: int = 0
    prefetch_factor: int = 2
    split_pct: float = 0.8
    split_method: str = "random"
    batch_size: int = 64
    shuffle: bool = True
    device: str = "cuda"
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    scheduler_name: Optional[str] = None
    scheduler_hparams: Dict[str, Any] = {}
    epochs: int = 100
    verbose: bool = False
    clip_grad_max_norm: float = 10.0
    checkpoint_log_every: int = 10
    plot_log_every: int = 10
    plot_n_samples: int = 10000
    plot_method: Optional[str] = None
    train_subsample_pct: float = 1.0
    valid_subsample_pct: float = 1.0
    use_wandb: bool = False
    final_shape: int = None

    inference_batch_size: int = 128


def main(cfg: SymmetricConv2dVAEConfig):
    # Initialize the model
    trainer = SymmetricConv2dVAETrainer(
        input_shape=cfg.input_shape,
        filters=cfg.filters,
        kernels=cfg.kernels,
        strides=cfg.strides,
        affine_widths=cfg.affine_widths,
        affine_dropouts=cfg.affine_dropouts,
        latent_dim=cfg.latent_dim,
        activation=cfg.activation,
        output_activation=cfg.output_activation,
        lambda_rec=cfg.lambda_rec,
        seed=cfg.seed,
        in_gpu_memory=cfg.in_gpu_memory,
        num_data_workers=cfg.num_data_workers,
        prefetch_factor=cfg.prefetch_factor,
        split_pct=cfg.split_pct,
        split_method=cfg.split_method,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        device=cfg.device,
        optimizer_name=cfg.optimizer_name,
        optimizer_hparams=cfg.optimizer_hparams,
        scheduler_name=cfg.scheduler_name,
        scheduler_hparams=cfg.scheduler_hparams,
        epochs=cfg.epochs,
        verbose=cfg.verbose,
        clip_grad_max_norm=cfg.clip_grad_max_norm,
        checkpoint_log_every=cfg.checkpoint_log_every,
        plot_log_every=cfg.plot_log_every,
        plot_n_samples=cfg.plot_n_samples,
        plot_method=cfg.plot_method,
        train_subsample_pct=cfg.train_subsample_pct,
        valid_subsample_pct=cfg.valid_subsample_pct,
        use_wandb=cfg.use_wandb,
    )

    print(trainer.model)

    # Load input data from HDF5 file
    if str(cfg.input_path)[-3:] == '.h5':
        with h5py.File(cfg.input_path) as f:
            contact_maps = f["contact_map"][...]
            scalars = {"rmsd": f["rmsd"][...]}
    # iterate over the directory
    else:
        contact_maps = []
        scalars = {'rmsd': []}
        for i in glob.glob(str(cfg.input_path) + '/*.h5'):
            with h5py.File(i, 'r') as f:
                contact_maps.extend(list(f["contact_map"][...]))
                scalars['rmsd'].extend(list(f["rmsd"][...]))

    print(f"Number of contact maps: {len(contact_maps)}")

    # Train model
    trainer.fit(
        X=contact_maps,
        scalars=scalars,
        output_path=cfg.output_path,
        checkpoint=cfg.checkpoint_path,
        final_shape=cfg.final_shape
    )

    pd.DataFrame(trainer.loss_curve_).to_csv(cfg.output_path / "loss.csv")

    # Generate latent embeddings in inference mode
    z, recon_vectors, loss, recon_loss, kld_loss, = trainer.predict(
        X=contact_maps, inference_batch_size=cfg.inference_batch_size, just_valid=True
    )

    np.save(cfg.output_path / "z.npy", z)
    np.save(cfg.output_path / "recon_vectors.npy", recon_vectors)

    print(f"Final loss on the full dataset is: {loss}, recon: {recon_loss}, kld: {kld_loss}")


if __name__ == "__main__":
    # Generate sample yaml
    # SymmetricConv2dVAEConfig().dump_yaml("symmetric_conv2d_vae_template.yaml")
    # exit()
    args = parse_args()
    cfg = SymmetricConv2dVAEConfig.from_yaml(args.config)
    main(cfg)
