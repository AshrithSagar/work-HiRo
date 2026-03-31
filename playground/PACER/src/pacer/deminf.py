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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from typingkit.core import TypedDict

from pacer.base import Actions, Demonstration, Demonstrations, States
from pacer.typings import DemoIndex, DimAction, DimState, Matrix, NumDemos, NumPoints

## ── DemInf ───────────────────────────────────────────────────────────────────

Score: TypeAlias = float


@dataclass
class DemInfEstimator:
    """Estimates `I(S; A) = H(S) + H(A) - H(S, A)` using the KSG Estimator."""

    k: int = 3

    def estimate(
        self,
        states: States[NumPoints, DimState],
        actions: Actions[NumPoints, DimAction],
    ) -> Score:
        """
        Calculates the Mutual Information score for a set of state-action pairs.
        Higher score => Higher quality demonstration.
        """

        states_np = states.numpy()
        actions_np = actions.numpy()

        # Ensure we have enough samples for k-NN
        n_samples = states.length
        if n_samples <= self.k:
            return 0.0

        # KSG Algorithm 1 Implementation
        # Reference: Kraskov et al. "Estimating mutual information." (2004).

        # Add small noise to prevent duplicate distances (jittering)
        states_np += 1e-10 * np.random.randn(*states_np.shape)
        actions_np += 1e-10 * np.random.randn(*actions_np.shape)

        # Combined space
        states_actions_np = np.hstack([states_np, actions_np])

        # Find distances to k-th neighbor in joint space
        nn_joint = NearestNeighbors(n_neighbors=self.k + 1, metric="chebyshev")
        nn_joint.fit(states_actions_np)
        distances, _ = nn_joint.kneighbors(states_actions_np)
        eps = distances[:, self.k]  # Distance to k-th neighbor

        # Count points within eps in marginal spaces
        def count_neighbors(data: Matrix[Any, Any], eps: Any) -> NDArray[Any]:
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
class DemInfScorer:
    """Utility to score entire demonstrations or sub-samples."""

    estimator: DemInfEstimator

    def score_demonstration(
        self, demo: Demonstration[NumPoints, DimState, DimAction]
    ) -> Score:
        """Score a single trajectory."""
        return self.estimator.estimate(demo.states, demo.actions)

    def score_demonstrations(
        self, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> TypedDict[NumDemos, DemoIndex, Score]:
        """Score all demonstrations in a collection."""
        scores = dict[DemoIndex, Score]()
        for demo in demonstrations:
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
