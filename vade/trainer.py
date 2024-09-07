import lightning as L
from torch import nn
import torch
from torch.optim.lr_scheduler import ExponentialLR
from .model import Encoder, Decoder
from .losses import mae_loss, KL_loss, contractive_loss, transf_invariant_loss, classification_loss


class VAE(L.LightningModule):
    def __init__(self, conv_filt, hidden, input_size, input_channels=3,
                 kl_beta=0.5, contractive_loss=False, rot_inv_loss=False,
                 learning_rate=1.e-3, lr_decay=0.98, optim_params={'name': 'adam', 'betas': (0.9, 0.99)},
                 decay_freq=5
                 ):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False

        self.encoder = Encoder(conv_filt, hidden, input_size, input_channels)
        self.decoder = Decoder(conv_filt, hidden[::-1], self.encoder.final_size, hidden[-1], input_channels)

    def forward(self, x):
        mu, sig, z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        '''
            Train the generator and discriminator on a single batch
        '''
        _, _, _, _, mean_loss = self.batch_step(batch)

        scheduler = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.hparams.decay_freq == 0:
            scheduler.step()

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

        return mean_loss['loss']

    def validation_step(self, batch, batch_idx):
        if self.hparams.contractive_loss:
            torch.set_grad_enabled(True)
        _, _, _, _, mean_loss = self.batch_step(batch)

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

        return mean_loss['loss']

    def batch_step(self, batch):
        img, labels = batch

        if self.hparams.contractive_loss:
            img.requires_grad_(True)
            img.retain_grad()

        mu, sig, z = self.encoder(img)
        gen = self.decoder(z)
        mup, sigp, zp = self.encoder(gen)

        loss = {}
        gen_img_loss = mae_loss(img, gen)
        kl_loss = self.hparams.kl_beta * KL_loss(mu, sig, mup)

        loss['gen_loss'] = gen_img_loss.item()
        loss['kl_loss'] = kl_loss.item()

        total_loss = gen_img_loss + kl_loss

        if self.hparams.contractive_loss:
            contr_loss = contractive_loss(z, img)
            loss['contr_loss'] = contr_loss.item()
            total_loss += contr_loss

        if self.hparams.rot_inv_loss:
            Linv, Lres = transf_invariant_loss(img, mu, self.encoder)
            loss['Linv'] = Linv.item()
            loss['Lres'] = Lres.item()
            total_loss += Linv + Lres

        return mu, sig, z, gen, {'loss': total_loss, **loss}

    def get_model_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate

        parameters = self.get_model_params()
        optim = self.hparams.optim_params.pop('name')
        if optim == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, **self.hparams.optim_params)
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(parameters, lr=learning_rate, **self.hparams.optim_params)
        elif optim == 'nadam':
            optimizer = torch.optim.NAdam(parameters, lr=learning_rate, **self.hparams.optim_params)
        else:
            raise ValueError(f'{self.hparams.optim_params["name"]} not implemented')

        lr_scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)

        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch",
                               "frequency": self.hparams.decay_freq}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class ClassVAE(VAE):
    def __init__(self, num_classes, *VAE_args, class_beta=10., **VAE_kwargs):
        super().__init__(*VAE_args, **VAE_kwargs)

        hidden = VAE_args[1] if 'hidden' not in VAE_kwargs else VAE_kwargs['hidden']

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.final_size * self.encoder.final_size * hidden[-1], 64),
            nn.InstanceNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mu, sig, z = self.encoder(x)
        return self.decoder(z), self.classifier(z)

    def batch_step(self, batch):
        img, label = batch
        mu, sig, z, gen, loss = super().batch_step(batch)

        pred_label = self.classifier(z)

        class_loss = self.hparams.class_beta * classification_loss(pred_label, label)

        loss['loss'] += class_loss

        return mu, sig, z, gen, {**loss, 'class_loss': class_loss.item()}

    def get_model_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
