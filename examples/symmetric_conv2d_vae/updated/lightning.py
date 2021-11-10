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

class CVAE(pl.LightningModule):
    def __init__(self, input_shape, input_path):
        super(CVAE, self).__init__()
        self.input_shape = input_shape
        self.input_path = input_path
        # self.config = config
        # sigmoid is part of BCE with logits loss
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                             in_channels=in_channels, out_channels=classes, init_features=32, pretrained=True)
        # self.model = smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        self.model = SymmetricConv2dVAE(input_shape=input_shape, filters=[64, 64, 64, 64], kernels=[5, 3, 3, 3],
                                        strides=[2, 2, 2, 2], latent_dim=10)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        with h5py.File(input_path) as f:
            contact_maps = np.array(f["contact_map"])
            scalars = {"rmsd": np.array(f["rmsd"])}

        print(f"Number of contact maps: {len(contact_maps)}")

        dataset = ContactMapDataset(data=contact_maps, shape=self.input_shape, scalars=scalars)
        self.train_loader, self.valid_loader = train_valid_split(
            dataset,
            0.8,
            "random",
            batch_size=64,
            shuffle=True,
            num_workers=10
        )

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.valid_loader

    def configure_optimizers(self):
        # auto finding learning rate
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # wandb.log({'learning_rate': self.learning_rate})
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
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
        kld_loss = self.model.kld_loss().half()
        recon_loss = self.criterion(recon_x.half(), x.half())
        loss = 1.0 * recon_loss + kld_loss
        # loss = recon_loss
        # only use when  on dp
        logs = {'train_loss': loss.detach().cpu()}
        self.log("training", logs)
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x = train_batch["X"].half()
        return {'forward': self.forward(x), 'expected': x}

    def test_step_end(self, outputs):
        _, recon_x = outputs['forward']
        # recon_x = recon_x.clamp(0, 1)
        x = outputs['expected']
        x = x.half()
        kld_loss = self.model.kld_loss().half()
        recon_loss = self.criterion(recon_x, x)
        loss = 1.0 * recon_loss + kld_loss
        # loss = recon_loss
        # only use when  on dp
        logs = {'test_loss': loss.detach().cpu()}
        self.log("test", logs)
        return {'loss': loss, 'test_logs': logs}


def lightning():
    torch.manual_seed(0)
    model = CVAE(input_shape=[1, 926, 926],
                 input_path='/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/gordon_bell/bba_deepdrive/chainA_h5_data/traj_segment_eq.2.1.h5')
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=5, gpus=1, auto_select_gpus=True, logger=wandb_logger, precision=16,
                         strategy=DDPPlugin(find_unused_parameters=False))
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    wandb.init(project='cvae', entity='mzvyagin')
    lightning()