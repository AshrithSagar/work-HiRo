"""
Train tiny β-VAEs for DemInf (state and action encoders)
Run this once before using DemInf.
"""
# tests/train_vaes.py

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deminf.deminf import BetaVAE
from pacer.testutils import get_demonstrations
from pacer.typings import npDType


def train_vae(
    data: np.ndarray,
    input_dim: int,
    latent_dim: int = 8,
    beta: float = 0.01,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> BetaVAE:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training VAE on device: {device}")

    model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, beta=beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(data.astype(npDType)))  # pyright: ignore[reportUnknownMemberType]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | Loss: {total_loss / len(loader):.6f}"
            )

    model.eval()
    return model


if __name__ == "__main__":
    demos = get_demonstrations(choice="FROM_LASA", pattern="GShape")

    # Collect all states and actions
    all_states = np.concatenate([demo.states.numpy() for demo in demos], axis=0)
    all_actions = np.concatenate([demo.actions.numpy() for demo in demos], axis=0)

    print(f"Training state VAE on {all_states.shape} data")
    state_vae = train_vae(
        all_states, input_dim=demos.state_dim, latent_dim=8, beta=0.005
    )

    print(f"\nTraining action VAE on {all_actions.shape} data")
    action_vae = train_vae(
        all_actions, input_dim=demos.action_dim, latent_dim=6, beta=0.005
    )

    # Save models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state_vae.state_dict(), os.path.join(ckpt_dir, "state_vae.pth"))
    torch.save(action_vae.state_dict(), os.path.join(ckpt_dir, "action_vae.pth"))

    print("\nVAEs trained and saved!")
