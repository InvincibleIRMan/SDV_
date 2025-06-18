import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularVAE(nn.Module):
    """Simple Variational Autoencoder for tabular data."""

    def __init__(self, input_dim, latent_dim=16, hidden_dims=(32, 32)):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(prev, latent_dim)
        self.logvar_layer = nn.Linear(prev, latent_dim)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.ReLU())
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def create_dataloader(df, categorical_columns, continuous_columns, batch_size=64):
    """Create a ``DataLoader`` handling categorical and continuous values.

    The categorical columns are one-hot encoded and concatenated with the
    continuous columns. The resulting tensor is returned in a ``DataLoader``.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('``df`` must be a pandas DataFrame')

    cat = pd.get_dummies(df[categorical_columns].astype('category'))
    cont = df[continuous_columns]
    data = pd.concat([cat, cont], axis=1).astype(np.float32).values

    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, epochs=100, lr=1e-3):
    """Train the VAE using the provided ``DataLoader``."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch, in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()


def sample(model, num_samples):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.mu_layer.out_features)
        samples = model.decoder(z)
    return samples.numpy()


if __name__ == '__main__':
    # Example usage with mixed categorical and continuous data
    df = pd.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C'], size=1000),
        'cat2': np.random.choice(['X', 'Y'], size=1000),
        'num1': np.random.normal(size=1000),
        'num2': np.random.uniform(-1, 1, size=1000),
    })

    dataloader = create_dataloader(
        df,
        categorical_columns=['cat1', 'cat2'],
        continuous_columns=['num1', 'num2'],
        batch_size=32,
    )

    input_dim = next(iter(dataloader))[0].shape[1]
    vae = TabularVAE(input_dim=input_dim, latent_dim=4)
    train(vae, dataloader, epochs=50)
    synthetic = sample(vae, 5)
    print(synthetic)
