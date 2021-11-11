import pytorch_lightning as pl
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAE
from mdlearn.data.utils import train_valid_split
from mdlearn.data.datasets.contact_map import ContactMapDataset
import h5py
import wandb
import torch
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import glob

NUM_DATA_WORKERS = 4

class CVAE(pl.LightningModule):
    def __init__(self, input_shape, input_path_list):
        super(CVAE, self).__init__()
        self.input_shape = input_shape
        # self.input_path = input_path
        # self.config = config
        # sigmoid is part of BCE with logits loss
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                             in_channels=in_channels, out_channels=classes, init_features=32, pretrained=True)
        # self.model = smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        self.model = SymmetricConv2dVAE(input_shape=input_shape, filters=[64, 64, 64, 32], kernels=[5, 5, 5, 5],
                                        strides=[2, 2, 2, 2], latent_dim=10, affine_widths=[64], activation="None")
        self.criterion = torch.nn.BCEWithLogitsLoss()

        contact_maps = []
        scalars = []
        for file in input_path_list:
            with h5py.File(file, "r") as f:
                contact_maps.extend(f["contact_map"][...])
                scalars.extend(f["rmsd"][...])
                # scalars = {"rmsd": np.array(f["rmsd"])}

        print(f"Number of contact maps: {len(contact_maps)}")
        scalars = {"rmsd": scalars}

        dataset = ContactMapDataset(data=contact_maps, shape=self.input_shape, scalars=scalars, pad=True)
        self.train_loader, self.valid_loader = train_valid_split(
            dataset,
            0.8,
            "random",
            batch_size=256,
            num_workers=NUM_DATA_WORKERS,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.valid_loader

    def configure_optimizers(self):
        # auto finding learning rate
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # wandb.log({'learning_rate': self.learning_rate})
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00001, weight_decay=1e-08)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        x = train_batch["X"].half()
        return {'forward': self.forward(x), 'expected': x}

    def training_step_end(self, outputs):
        _, recon_x = outputs['forward']
        # recon_x = recon_x.clamp(0, 1)
        x = outputs['expected']
        x = x
        kld_loss = self.model.kld_loss().float()
        recon_loss = self.criterion(recon_x.half(), x.half()).float()
        loss = 1.0 * recon_loss + kld_loss
        # loss = recon_loss
        # only use when  on dp
        logs = {'train_loss': loss.detach().cpu(), 'recon_loss': recon_loss.detach().cpu(),
                'kld_loss': kld_loss.detach().cpu()}
        self.log("training", logs)
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x = test_batch["X"].half()
        return {'forward': self.forward(x), 'expected': x}

    def test_step_end(self, outputs):
        _, recon_x = outputs['forward']
        # recon_x = recon_x.clamp(0, 1)
        x = outputs['expected']
        x = x.half()
        kld_loss = self.model.kld_loss().float()
        recon_loss = self.criterion(recon_x, x).float()
        loss = 1.0 * recon_loss + kld_loss
        # loss = recon_loss
        # only use when  on dp
        logs = {'test_loss': loss.detach().cpu(), 'recon_loss': recon_loss.detach().cpu(),
                'kld_loss': kld_loss.detach().cpu()}
        self.log("test", logs)
        return {'loss': loss, 'test_logs': logs}


def lightning():
    torch.set_num_threads(NUM_DATA_WORKERS)
    torch.manual_seed(0)
    input_path_list = glob.glob('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/gordon_bell/anda_newsim_7egq_segmentA/chainA_subset/*.h5')
    model = CVAE(input_shape=[1, 928, 928],
                 input_path_list=input_path_list)
                 # input_path='/homes/mzvyagin/gordon_bell_processing/anda_newsim_7egq_segmentA/traj_segment_eq.2.10.h5')
    wandb_logger = WandbLogger(project="cvae", entity="mzvyagin", group="ddp")
    trainer = pl.Trainer(max_epochs=5, gpus=4, auto_select_gpus=True, logger=wandb_logger, precision=16, num_nodes=1,
                         strategy=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    # wandb.init(project='cvae', entity='mzvyagin', group="ddp")
    torch.backends.cudnn.benchmark = False
    lightning()