#!/usr/bin/env python3
"""Demonstration of a simple Tabular VAE for mixed data.

This script builds a small toy dataset containing continuous and
categorical features, preprocesses them, defines a Variational
Autoencoder that embeds the categorical inputs, and runs a short
training loop. It can be used as a quick tutorial to understand how to
model tabular data with a VAE.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------------------------------------------
# 1. Create a toy tabular dataset
# ----------------------------------------------------------------------


def create_toy_data(n=200, seed=42):
    """Return a small synthetic dataset with mixed feature types."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'age': rng.integers(20, 65, size=n),
        'salary': (50_000 + 12_000 * rng.standard_normal(size=n)).round(),
        'gender': rng.choice(['Male', 'Female'], size=n),
        'department': rng.choice(['sales', 'engineering', 'finance'], size=n),
    })


df = create_toy_data()
categorical = ['gender', 'department']
continuous = ['age', 'salary']

# ----------------------------------------------------------------------
# 2. Preprocess features
# ----------------------------------------------------------------------

scaler = StandardScaler().fit(df[continuous])
cont_arr = scaler.transform(df[continuous]).astype(np.float32)

encoders = {col: LabelEncoder().fit(df[col]) for col in categorical}
cat_arr = np.stack(
    [encoders[c].transform(df[c]) for c in categorical],
    axis=1,
).astype(np.int64)

cont_train, cont_val, cat_train, cat_val = train_test_split(
    cont_arr,
    cat_arr,
    test_size=0.2,
    random_state=42,
)


def yield_batches(arr1, arr2, batch_size):
    """Yield mini-batches from two aligned arrays."""
    for i in range(0, len(arr1), batch_size):
        yield arr1[i : i + batch_size], arr2[i : i + batch_size]


# ----------------------------------------------------------------------
# 3. VAE model
# ----------------------------------------------------------------------


class MixedTabularVAE(nn.Module):
    """Variational Autoencoder for mixed continuous and categorical data."""

    def __init__(self, cont_dim, cat_dims, emb_sizes, latent_dim=8):
        """Initialize the network layers."""
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_size) for cat_dim, emb_size in zip(cat_dims, emb_sizes)
        ])
        input_dim = cont_dim + sum(emb_sizes)

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, cont_x, cat_x):
        """Encode inputs into latent parameters."""
        emb = [emb_layer(cat_x[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([cont_x] + emb, dim=1)
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample from the latent space using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent samples back to the input space."""
        h = torch.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, cont_x, cat_x):
        """Run a forward pass through the network."""
        mu, logvar = self.encode(cont_x, cat_x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ----------------------------------------------------------------------
# 4. Training setup
# ----------------------------------------------------------------------

cont_dim = len(continuous)
cat_dims = [len(encoders[c].classes_) for c in categorical]
emb_sizes = [min(50, (dim + 1) // 2) for dim in cat_dims]

model = MixedTabularVAE(cont_dim, cat_dims, emb_sizes, latent_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_fn(recon, target, mu, logvar):
    """Return the VAE loss for a batch."""
    mse = nn.MSELoss(reduction='sum')(recon, target)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


# ----------------------------------------------------------------------
# 5. Training loop
# ----------------------------------------------------------------------

n_epochs = 30
batch_size = 32


def main() -> None:
    """Train the VAE on the toy dataset and report losses."""
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for xb_cont, xb_cat in yield_batches(cont_train, cat_train, batch_size):
            xb_cont = torch.tensor(xb_cont)
            xb_cat = torch.tensor(xb_cat)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb_cont, xb_cat)
            emb_inputs = [model.embeddings[i](xb_cat[:, i]) for i in range(len(model.embeddings))]
            x = torch.cat([xb_cont] + emb_inputs, dim=1)
            loss = loss_fn(recon, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_cont, xb_cat in yield_batches(cont_val, cat_val, batch_size):
                xb_cont = torch.tensor(xb_cont)
                xb_cat = torch.tensor(xb_cat)
                recon, mu, logvar = model(xb_cont, xb_cat)
                emb_inputs = [
                    model.embeddings[i](xb_cat[:, i]) for i in range(len(model.embeddings))
                ]
                x = torch.cat([xb_cont] + emb_inputs, dim=1)
                val_loss += loss_fn(recon, x, mu, logvar).item()

        print(  # noqa: T201
            f'Epoch {epoch + 1:02d} | '
            f'Train Loss: {train_loss / len(cont_train):.4f} | '
            f'Val Loss: {val_loss / len(cont_val):.4f}',
        )


if __name__ == '__main__':
    main()
