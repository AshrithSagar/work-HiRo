"""
DemInf
=======
Implementation follows the following paper from
Joey Hejna, Suvir Mirchandani, et. al., RSS 2025:
"Robot Data Curation with Mutual Information Estimators"
https://www.roboticsproceedings.org/rss21/p023.pdf
https://arxiv.org/abs/2502.08623
"""
# src/deminf/deminf.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, override

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from typingkit.core import TypedDict

from pacer.base import Actions, Demonstration, Demonstrations, States
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    Matrix,
    NumDemos,
    NumPoints,
    npDType,
)

## ── DemInf ───────────────────────────────────────────────────────────────────

Score: TypeAlias = float  # Sc(tau)
# Paper uses S for both states and scores, so we'll stick to
# the notation S for States, and Sc for Scores hereinafter.


class MutualInformationEstimator(Protocol):
    """Interface for any mutual information estimator between states and actions."""

    def estimate(
        self,
        states: States[NumPoints, DimState],
        actions: Actions[NumPoints, DimAction],
    ) -> Score:
        """
        Estimate mutual information I(S; A) for the given state-action pairs.
        Higher value indicates higher quality (more informative) data.
        """
        ...


@dataclass
class KSGEstimator(MutualInformationEstimator):
    """
    Kraskov-Stögbauer-Grassberger (KSG) MI estimator.\\
    This follows the classic KSG Algorithm 1 (Kraskov et al., 2004).
    """

    k: int = 3
    jitter: float = 1e-10

    @override
    def estimate(
        self,
        states: States[NumPoints, DimState],
        actions: Actions[NumPoints, DimAction],
    ) -> Score:
        """Calculate MI score using KSG estimator."""

        states_np = states.numpy()
        actions_np = actions.numpy()

        # Ensure we have enough samples for k-NN
        n_samples = states.length
        if n_samples <= self.k:
            return Score(0.0)

        # Add small noise to prevent duplicate distances (jittering)
        if self.jitter > 0.0:
            states_np += 1e-10 * np.random.randn(*states_np.shape)
            actions_np += 1e-10 * np.random.randn(*actions_np.shape)

        # Joint space for k-NN
        joint_np = np.hstack([states_np, actions_np])

        # Find distances to k-th neighbor in joint space (using Chebyshev / max-norm)
        nn_joint = NearestNeighbors(n_neighbors=self.k + 1, metric="chebyshev")
        nn_joint.fit(joint_np)
        distances, _ = nn_joint.kneighbors(joint_np)
        eps = distances[:, self.k]  # Distance to k-th neighbor

        # Count points within eps in marginal spaces
        def count_neighbors(data: Matrix[Any, Any], eps: Any) -> NDArray[np.int64]:
            nn = NearestNeighbors(metric="chebyshev")
            nn.fit(data)
            # Find points within radius eps (strictly less than)
            # The slightly smaller radius is to ensure we don't include the k-th neighbor itself
            count = nn.radius_neighbors(data, radius=eps - 1e-15, return_distance=False)
            return np.array([len(c) for c in count])

        nx = count_neighbors(states_np, eps)
        ny = count_neighbors(actions_np, eps)

        # MI: psi(k) - <psi(nx + 1) + psi(ny + 1)> + psi(N)
        mi = (
            digamma(self.k)
            - np.mean(digamma(nx + 1) + digamma(ny + 1))
            + digamma(n_samples)
        )

        return Score(max(0.0, mi))


class RepresentationEncoder(Protocol):
    """
    Protocol for encoders that produce low-dimensional representations
    (e.g., VAE latents for states or actions).
    """

    def encode(self, x: NDArray[npDType]) -> NDArray[npDType]:
        """Encode batch: (N, dim) -> (N, latent_dim)"""
        ...


@dataclass
class DemInfRepresentationEstimator:
    """
    DemInf: VAE embeddings + KSG estimator.
    - Uses separate VAE latents `zs`, `za` for states and actions
    - Max-norm joint distance: max( L2(zs), L2(za) )
    - Per-sample contribution: `-digamma(n_zs + 1) - digamma(n_za + 1)`
    - Clipping + randomized batching
    """

    state_encoder: RepresentationEncoder
    action_encoder: RepresentationEncoder

    k_values: Sequence[int] = field(default_factory=lambda: [5, 6, 7])
    n_shuffles: int = 4
    batch_size: int = 1024
    clip_percentiles: tuple[float, float] = (1.0, 99.0)

    def _collect_all(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> tuple[NDArray[npDType], NDArray[npDType]]:
        """Collect all states and actions from demonstrations"""
        states_list = [demo.states.numpy() for demo in demos]
        actions_list = [demo.actions.numpy() for demo in demos]
        return np.concatenate(states_list, axis=0), np.concatenate(actions_list, axis=0)

    def _embed(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> tuple[NDArray[npDType], NDArray[npDType]]:
        all_states, all_actions = self._collect_all(demos)
        zs = self.state_encoder.encode(all_states)
        za = self.action_encoder.encode(all_actions)
        return zs, za

    def _per_sample_contributions(
        self, zs: NDArray[npDType], za: NDArray[npDType]
    ) -> NDArray[npDType]:
        """Compute per-sample MI contributions using max-norm KSG."""
        M = zs.shape[0]
        assert za.shape[0] == M

        all_contrib = np.zeros(M, dtype=npDType)

        for _ in range(self.n_shuffles):
            idx = np.random.permutation(M)
            zs_shuf = zs[idx]
            za_shuf = za[idx]

            for start in range(0, M, self.batch_size):
                end = min(start + self.batch_size, M)
                zs_b = zs_shuf[start:end]
                za_b = za_shuf[start:end]
                Mb = zs_b.shape[0]

                # L2 distances per space
                dist_s = np.linalg.norm(zs_b[:, None] - zs_b[None, :], axis=-1)
                dist_a = np.linalg.norm(za_b[:, None] - za_b[None, :], axis=-1)

                # Joint max-norm
                joint_dist = np.maximum(dist_s, dist_a)
                np.fill_diagonal(joint_dist, np.inf)

                contrib_batch = np.zeros(Mb)

                for k in self.k_values:
                    # k-th nearest neighbor distance in joint space
                    rho_k = np.sort(joint_dist, axis=1)[:, k]

                    # Count neighbors in marginal spaces (exclude self)
                    n_zs = np.sum(dist_s <= rho_k[:, None], axis=1) - 1
                    n_za = np.sum(dist_a <= rho_k[:, None], axis=1) - 1

                    contrib_batch += -digamma(n_zs + 1) - digamma(n_za + 1)

                contrib_batch /= len(self.k_values)
                all_contrib[idx[start:end]] += contrib_batch

        scores = all_contrib / self.n_shuffles

        # Clip outliers
        low, high = np.percentile(scores, self.clip_percentiles)
        return np.clip(scores, low, high, dtype=npDType)

    def score_demonstrations(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> dict[DemoIndex, Score]:
        if len(demos) == 0:
            return {}
        zs, za = self._embed(demos)
        per_sample_scores = self._per_sample_contributions(zs, za)

        scores: dict[DemoIndex, Score] = {}
        idx = 0
        for demo in demos:
            T = len(demo)
            scores[demo.index] = Score(per_sample_scores[idx : idx + T].mean())
            idx += T
        return scores


@dataclass
class DemInfEstimator:
    state_encoder: RepresentationEncoder
    action_encoder: RepresentationEncoder
    k_values: Sequence[int] = field(default_factory=lambda: [5, 6, 7])
    ##
    core: DemInfRepresentationEstimator = field(init=False)

    def __post_init__(self) -> None:
        self.core = DemInfRepresentationEstimator(
            state_encoder=self.state_encoder,
            action_encoder=self.action_encoder,
            k_values=self.k_values,
        )

    def score_demonstrations(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> dict[DemoIndex, Score]:
        return self.core.score_demonstrations(demos)


@dataclass
class Scorer:
    """Utility to score entire demonstrations or sub-samples."""

    estimator: MutualInformationEstimator

    def score_demonstration(
        self, demo: Demonstration[NumPoints, DimState, DimAction]
    ) -> Score:
        """Score a single trajectory."""
        return self.estimator.estimate(demo.states, demo.actions)

    def score_demonstrations(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> TypedDict[NumDemos, DemoIndex, Score]:
        """Score all demonstrations in a collection."""
        scores = dict[DemoIndex, Score]()
        for demo in demos:
            scores[demo.index] = self.score_demonstration(demo)
        return TypedDict[NumDemos, DemoIndex, Score](scores)

    def rank_scores(
        self, scores: TypedDict[NumDemos, DemoIndex, Score]
    ) -> TypedDict[NumDemos, DemoIndex, Score]:
        """Rank scores by quality (highest DemInf first)."""
        return TypedDict[NumDemos, DemoIndex, Score](
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )


@dataclass
class BatchScorer:
    """Utility to score and rank demonstrations."""

    estimator: DemInfRepresentationEstimator

    def score_demonstrations(
        self, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> TypedDict[NumDemos, DemoIndex, Score]:
        raw = self.estimator.score_demonstrations(demos)
        return TypedDict[NumDemos, DemoIndex, Score](raw)

    def rank_scores(
        self, scores: TypedDict[NumDemos, DemoIndex, Score]
    ) -> TypedDict[NumDemos, DemoIndex, Score]:
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return TypedDict[NumDemos, DemoIndex, Score](dict(sorted_items))


@dataclass
class MLPRepresentationEncoder(RepresentationEncoder):
    """Simple concrete implementation using a PyTorch MLP (e.g., VAE encoder head)."""

    vae: BetaVAE
    device: torch.device = torch.device("cpu")

    @override
    def encode(self, x: NDArray[npDType]) -> NDArray[npDType]:
        self.vae.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(npDType)).to(self.device)  # pyright: ignore[reportUnknownMemberType]
            tensor = torch.from_numpy(x).to(self.device)  # pyright: ignore[reportUnknownMemberType]
            z = self.vae.encode(tensor)
            return z.cpu().numpy().astype(npDType)


class SimpleVAEEncoder(nn.Module):
    """Tiny β-VAE encoder."""

    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.net: nn.Module = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),  # mu + logvar
        )
        self.latent_dim: int = latent_dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization


class BetaVAE(nn.Module):
    """Simple β-VAE for states or actions (paper uses lightweight MLPs + isotropic Gaussian prior)."""

    def __init__(self, input_dim: int, latent_dim: int = 8, beta: float = 0.01) -> None:
        super().__init__()
        self.beta: float = beta
        self.latent_dim: int = latent_dim

        self.encoder: nn.Module = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),  # mu + logvar
        )

        self.decoder: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        """Return latent sample (for DemInf encoder)."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss(self, x: Tensor) -> Tensor:
        """β-VAE loss: reconstruction + β * KL."""
        x_recon, mu, logvar = self.forward(x)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

        # KL divergence (closed form for Gaussian)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_loss


## ─────────────────────────────────────────────────────────────────────────────
