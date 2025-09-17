import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_layers, dec_layers, output_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers

        # encoder
        enc_modules = []
        for i in range(enc_layers):
            if i == 0:
                enc_modules.append(nn.Linear(input_dim, self.latent_dim))
            else:
                enc_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            enc_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_modules)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # decoder
        dec_modules = []
        for i in range(dec_layers):
            if i == 0:
                dec_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            else:
                dec_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            dec_modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_modules)
        self.fc_output = nn.Linear(self.latent_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)

        nn.init.xavier_normal_(self.fc_mu.weight.data)
        nn.init.xavier_normal_(self.fc_logvar.weight.data)
        nn.init.xavier_normal_(self.fc_output.weight.data)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        reconstructed = self.fc_output(z)
        return reconstructed

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z

# KL 散度损失
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
