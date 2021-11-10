import pytorch_lightning as pl
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAE
from mdlearn.data.utils import train_valid_split
from mdlearn.data.datasets.contact_map import ContactMapDataset
import h5py
import wandb
import torch


class CVAE(pl.LightningModule):
    def __init__(self, input_shape, input_path):
        super(CVAE, self).__init__()
        self.config = config
        # sigmoid is part of BCE with logits loss
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                             in_channels=in_channels, out_channels=classes, init_features=32, pretrained=True)
        # self.model = smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=in_channels, classes=classes)
        self.model = SymmetricConv2dVAE(input_shape=input_shape)

        with h5py.File(input_path) as f:
            contact_maps = f["contact_map"][...]
            scalars = {"rmsd": f["rmsd"][...]}

        print(f"Number of contact maps: {len(contact_maps)}")

        dataset = ContactMapDataset(X, self.input_shape, scalars)
        self.train_loader, self.valid_loader = train_valid_split(
            dataset,
            0.8,
            "partition",
            batch_size=4,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
            drop_last=True,
            pin_memory=False,
        )

        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = smp.losses.DiceLoss(mode="binary")
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.learning_rate = 0.0
        self.accuracy = torchmetrics.Accuracy()
        self.iou = torchmetrics.IoU(num_classes=2)
        self.train_set, self.valid_set, self.test_set = pt_gis_train_test_split(image_type=image_type)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.valid_loader

    def configure_optimizers(self):
        # auto finding learning rate
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # wandb.log({'learning_rate': self.learning_rate})
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        return {'forward': self.forward(train_batch), 'expected': train_batch}

    def training_step_end(self, outputs):
        _, recon_x = outputs['forward']
        kld_loss = self.model.kld_loss()
        recon_loss = self.model.recon_loss(x, recon_x)
        loss = 1.0 * recon_loss + kld_loss
        # only use when  on dp
        logs = {'train_loss': loss.detach().cpu(), "train_recon_loss":recon_loss.detach().cpu(),
                "train_kld_loss": kld_loss.detach().cpu()}
        self.log("training", logs)
        return {'train_loss': loss, 'logs': logs, 'train_recon': recon_loss, 'train_kld': kld}

    def test_step(self, test_batch, batch_idx):
        return {'forward': self.forward(test_batch), 'expected': test_batch}

    def test_step_end(self, outputs):
        _, recon_x = outputs['forward']
        kld_loss = self.model.kld_loss()
        recon_loss = self.model.recon_loss(x, recon_x)
        loss = 1.0 * recon_loss + kld_loss
        # only use when  on dp
        logs = {'test_loss': loss.detach().cpu(), "test_recon_loss": recon_loss.detach().cpu(),
                "test_kld_loss": kld_loss.detach().cpu()}
        self.log("test", logs)
        return {'test_loss': loss, 'test_logs': logs, 'test_recon': recon_loss, 'test_kld': kld}


def lightning():
    torch.manual_seed(0)
    model = CVAE(input_shape=[1, 926, 926],
                 input_path='/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/gordon_bell/bba_deepdrive/chainA_h5_data'
                            '/traj_segment_eq.2.1.h5')
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=5, gpus=1, auto_select_gpus=True, logger=wandb_logger)
                         # callbacks=[EarlyStopping(monitor="val_accuracy")])
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    wandb.init(project='cvae', entity='mzvyagin')
    lightning()