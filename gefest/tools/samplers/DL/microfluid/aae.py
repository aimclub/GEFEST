import torch
import time

from torch import nn


class AAE(nn.Module):

    ################################
    # Adversarial Auto Encoder model
    ################################

    def __init__(self, Encoder, Decoder, Discriminator, hidden_dim, conv_dims, n_layers, device):
        """
        Creating AAE model
        :param Encoder: Encoder model
        :param Decoder: Decoder model
        :param Discriminator: Discriminator model
        :param hidden_dim: (Int) Dimension of hidden space
        :param conv_dims: (Int) List of channels dimensional through conv layers
        :param n_layers: (Int) Number of layers in discriminator
        :param device: Current working device
        """
        super(AAE, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(hidden_dim, conv_dims, device).to(device)
        self.decoder = Decoder(hidden_dim, conv_dims, device).to(device)
        self.discriminator = Discriminator(hidden_dim, n_layers, device).to(device)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        self.threshold = torch.Tensor([0.8])

    def __call__(self, x):
        """
        :param x: (Tensor) [B x C x W x H]
        :return: Tuple(Tensor) Return reconstruction object and disc probability
        """
        latent_sample = self.encoder(x)
        decoded_sample = self.decoder(latent_sample)
        probab_sample = self.discriminator(latent_sample)

        out = (decoded_sample, probab_sample)

        return out

    def recon_loss(self, x, recon_object):
        """
        Calculating reconstruction loss based on formula

        L_recon = |E_q {log p(x|z)}, where p(x|z) = |N(x | decoder(encoder(x)), I)

        :param x: (Tensor) [B x C x W x H] True object
        :param recon_object: (Tensor) [B x C x W x H]
        :return: (Float)
        """

        total_loss = self.mse(recon_object, x)

        return total_loss

    def encoder_reg(self, d_z):
        """
        Calculating encoder regularization based on formula

        L_encoder_reg = |E_q {-log D(z)}, where z ~ q(z|x) or z = encoder(x)

        :param d_z: (Tensor) [B x 1]
        :return: (Float)
        """
        batch_size = d_z.size(dim=0)

        ones = torch.ones(size=(batch_size, 1)).to(self.device)

        encoder_reg = self.bce(d_z, ones)

        return encoder_reg

    def discr_reg(self, d_z, d_prior):
        """
        Calculating discriminator loss based on formula

        L_discr = |E_q {-log (1 - D(z*))} + |E_p {-log D(z)},
        where z* ~ q(z*|x), z ~ p(z)

        :param d_z: (Tensor) [B x 1]
        :param d_prior: (Tensor) [B x 1]
        :return: (Float)
        """
        batch_size = d_z.size(dim=0)

        ones = torch.ones(size=(batch_size, 1)).to(self.device)
        zeros = torch.zeros(size=(batch_size, 1)).to(self.device)

        discr_reg = 0.5 * (self.bce(d_prior, ones) + self.bce(d_z, zeros))

        return discr_reg

    def get_losses(self, x):

        """
        Get all losses for calculating reconstruction and regularization phases
        :param x: (Tensor) [B x C x W x H]
        :return: (Float)
        """

        batch_size = x.size(dim=0)

        z_sample = self.encoder(x)
        d_z_sample = self.discriminator(z_sample)

        z_sample_1 = self.encoder(x)
        d_z_sample_1 = self.discriminator(z_sample_1)

        recon_object = self.decoder(z_sample)

        # Standard normal distribution was taken as prior
        z_prior = torch.randn(size=(batch_size, self.hidden_dim)).to(self.device)
        d_z_prior = self.discriminator(z_prior)

        recon_loss = self.recon_loss(x, recon_object)
        discrim_reg = self.discr_reg(d_z_sample, d_z_prior)
        encoder_reg = self.encoder_reg(d_z_sample_1)

        return recon_loss, discrim_reg, encoder_reg

    def fit(self, trainloader, testloader, epochs):
        """
        Optimizing VAE/GAN model
        :param trainloader: (Dataloader) Train dataloader
        :param testloader: (Dataloader) Test dataloader
        :param epochs: (Int) Number of epochs
        :return: (dict) History of losses
        """
        params = {
            'ae': list(self.decoder.parameters()) + list(self.encoder.parameters()),
            'encoder': list(self.encoder.parameters()),
            'discriminator': list(self.discriminator.parameters())
        }

        ae_optim = torch.optim.Adam(params=params['ae'], lr=1e-3)
        en_optim = torch.optim.Adam(params=params['encoder'], lr=1e-4)
        dis_optim = torch.optim.Adam(params=params['discriminator'], lr=1e-4)

        print('opt=%s(lr=%f), epochs=%d, device=%s,\n'
              'en loss \u2193, gen loss \u2193, dis loss \u2191' % \
              (type(en_optim).__name__,
               en_optim.param_groups[0]['lr'], epochs, self.device))

        history = {'loss': [], 'val_loss': []}

        train_len = len(trainloader.dataset)
        test_len = len(testloader.dataset)

        for epoch in range(epochs):
            start_time = time.time()

            #################
            # TRAINING PART
            # ###############

            for i, batch_samples in enumerate(trainloader):
                batch_samples = batch_samples['image'].to(self.device)

                #######################
                # RECONSTRUCTION PHASE
                #######################
                z_sample = self.encoder(batch_samples)
                recon_object = self.decoder(z_sample)
                recon_loss = self.recon_loss(batch_samples, recon_object)

                ae_optim.zero_grad()

                recon_loss.backward()
                ae_optim.step()

                #######################
                # REGULARIZATION PHASE
                #######################

                # Discriminator update
                z_sample = self.encoder(batch_samples)
                d_z_sample = self.discriminator(z_sample)

                # Standard normal distribution was taken as prior
                z_prior = torch.normal(mean=0, std=1, size=(batch_samples.size(dim=0), self.hidden_dim)).to(self.device)
                d_z_prior = self.discriminator(z_prior)

                discrim_reg = self.discr_reg(d_z_sample, d_z_prior)

                dis_optim.zero_grad()

                discrim_reg.backward()
                dis_optim.step()

                # Encoder (generator) update
                z_sample = self.encoder(batch_samples)
                d_z_sample = self.discriminator(z_sample)

                encoder_reg = self.encoder_reg(d_z_sample)

                en_optim.zero_grad()

                encoder_reg.backward()
                en_optim.step()

                end_time = time.time()
                work_time = end_time - start_time

                print('Epoch/batch %3d/%3d \n'
                      'recon_loss %5.5f, discr_reg %5.5f, encoder_reg %5.5f \n'
                      'batch time %5.2f sec' % \
                      (epoch + 1, i,
                       recon_loss.item(), discrim_reg.item(), encoder_reg.item(),
                       work_time))

            #######################
            # VALIDATION PART
            #######################
            val_recon_loss = 0.0
            val_gen_loss = 0.0
            val_dis_loss = 0.0

            with torch.no_grad():
                for i, batch_samples in enumerate(testloader):
                    batch_samples = batch_samples['image'].to(self.device)

                    recon_loss, discrim_reg, encoder_reg = self.get_losses(batch_samples)

                    val_recon_loss += recon_loss.item() * batch_samples.size(dim=0)
                    val_gen_loss += encoder_reg.item() * batch_samples.size(dim=0)
                    val_dis_loss += discrim_reg.item() * batch_samples.size(dim=0)

            val_recon_loss = val_recon_loss / test_len
            val_gen_loss = val_gen_loss / test_len
            val_dis_loss = val_dis_loss / test_len

            if epoch + 1:
                print('Epoch %3d/%3d \n'
                      'val_en_loss %5.5f, val_gen_loss %5.5f, val_dis_loss %5.5f \n' % \
                      (epoch + 1, epochs,
                       val_recon_loss, val_gen_loss, val_dis_loss))

            history['val_loss'].append(val_recon_loss + val_gen_loss + val_dis_loss)

        return history
