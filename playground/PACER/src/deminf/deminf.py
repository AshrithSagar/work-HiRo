"""
DemInf
=======
Implementation follows the following paper from
Joey Hejna, Suvir Mirchandani, et. al., RSS 2025:
"Robot Data Curation with Mutual Information Estimators"
https://www.roboticsproceedings.org/rss21/p023.pdf
https://arxiv.org/abs/2502.08623
"""
# src/pacer/deminf.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, override

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from typingkit.core import TypedDict

from pacer.base import Actions, Demonstration, Demonstrations, States
from pacer.typings import DemoIndex, DimAction, DimState, Matrix, NumDemos, NumPoints

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


## ─────────────────────────────────────────────────────────────────────────────
