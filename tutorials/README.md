# TVAE Tutorial

This directory contains a short tutorial on training a Variational
Autoencoder (VAE) for mixed tabular data. The script,
`TVAE_tutrorial.py`, generates a toy dataset with two numeric columns
(`age`, `salary`) and two categorical columns (`gender`, `department`).
It then standardizes the numeric features, integer-encodes the
categorical features, embeds them inside the model, and trains a small
VAE using PyTorch.

Run the example with:

```bash
python TVAE_tutrorial.py
```

It will print the training and validation losses for a few epochs while
fitting the model. You can also explore the same code interactively in
`TVAE_tutrorial.ipynb`.
