from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


@dataclass
class GMMParams:
    pi: np.ndarray  # shape (K,)
    mu: np.ndarray  # shape (K, D)
    sigma: np.ndarray  # shape (K, D, D)


class GMM:
    def __init__(self, k: int):
        self.k = k

    def fit(
        self, data: np.ndarray, max_iter: int = 100, tol: float = 1e-6
    ) -> GMMParams:
        """
        Fit GMM to data using EM algorithm.
        data: shape (N, D)
        """
        n, d = data.shape
        # Initialize
        pi = np.ones(self.k) / self.k
        idx = np.random.choice(n, self.k, replace=False)
        mu = data[idx]
        sigma = np.array([np.eye(d) for _ in range(self.k)])
        prev_loglik = -np.inf

        for _ in range(max_iter):
            # E-step
            probs = np.zeros((n, self.k))
            for j in range(self.k):
                probs[:, j] = pi[j] * multivariate_normal.pdf(
                    data, mean=mu[j], cov=sigma[j]
                )
            loglik = np.sum(np.log(np.sum(probs, axis=1) + 1e-10))
            probs /= np.sum(probs, axis=1, keepdims=True) + 1e-10

            # M-step
            nk = np.sum(probs, axis=0)
            pi = nk / n
            mu = np.dot(probs.T, data) / (nk[:, np.newaxis] + 1e-10)
            for j in range(self.k):
                diff = data - mu[j]
                sigma[j] = np.dot((probs[:, j][:, np.newaxis] * diff).T, diff) / (
                    nk[j] + 1e-10
                )
                sigma[j] += 1e-6 * np.eye(d)  # Regularization

            # Check convergence
            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik

        return GMMParams(pi, mu, sigma)

    def get_gamma(self, params: GMMParams, x: np.ndarray) -> np.ndarray:
        """
        Compute mixing weights gamma_k(x)
        """
        probs = np.zeros(params.pi.shape[0])
        for j in range(len(probs)):
            probs[j] = params.pi[j] * multivariate_normal.pdf(
                x, params.mu[j], params.sigma[j]
            )
        probs /= probs.sum() + 1e-10
        return probs


class LPVDS:
    def __init__(
        self,
        gmm_params: GMMParams,
        A: np.ndarray,
        x_star: np.ndarray,
        P: np.ndarray | None = None,
    ):
        """
        A: shape (K, D, D)
        """
        self.gmm_params = gmm_params
        self.A = A
        self.x_star = x_star
        self.P = P  # Lyapunov matrix, optional

    def compute_velocity(self, x: np.ndarray) -> np.ndarray:
        gamma = GMM(self.gmm_params.pi.size).get_gamma(self.gmm_params, x)
        dot_x = np.zeros_like(x)
        for j in range(gamma.size):
            dot_x += gamma[j] * (self.A[j] @ (x - self.x_star))
        return dot_x


class EMP:
    def __init__(
        self,
        k: int,
        demo_positions: np.ndarray,
        demo_velocities: np.ndarray,
        x_star: np.ndarray,
    ):
        """
        demo_positions: N x D
        demo_velocities: N x D
        x_star: D
        """
        self.k = k
        self.demo_positions = demo_positions
        self.demo_velocities = demo_velocities
        self.x_star = x_star
        self.d = demo_positions.shape[1]
        self.gmm = GMM(k)
        self.gmm_params = self.gmm.fit(demo_positions)
        self.gmm_params = self._order_components()
        self.A = self._learn_ds_params()
        self.P = self._learn_lyapunov()
        self.policy = LPVDS(self.gmm_params, self.A, x_star, self.P)

    def _order_components(self) -> GMMParams:
        """
        Order GMM components along the trajectory using weighted time indices.
        """
        n = self.demo_positions.shape[0]
        times = np.arange(n) / (n - 1)  # Normalized time
        probs = np.zeros((n, self.k))
        for j in range(self.k):
            probs[:, j] = self.gmm_params.pi[j] * multivariate_normal.pdf(
                self.demo_positions, self.gmm_params.mu[j], self.gmm_params.sigma[j]
            )
        probs /= probs.sum(1, keepdims=True)
        weighted_times = np.dot(probs.T, times) / probs.sum(0)
        order = np.argsort(weighted_times)
        return GMMParams(
            self.gmm_params.pi[order],
            self.gmm_params.mu[order],
            self.gmm_params.sigma[order],
        )

    def _learn_ds_params(self) -> np.ndarray:
        """
        Learn A_k using least squares (unconstrained).
        """
        delta_x = self.demo_positions - self.x_star
        n = delta_x.shape[0]
        A = np.zeros((self.k, self.d, self.d))
        for out_dim in range(self.d):
            y = self.demo_velocities[:, out_dim]
            X = np.zeros((n, self.k * self.d))
            for i in range(n):
                gamma = self.gmm.get_gamma(self.gmm_params, self.demo_positions[i])
                for j in range(self.k):
                    X[i, j * self.d : (j + 1) * self.d] = gamma[j] * delta_x[i]
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            for j in range(self.k):
                A[j, out_dim, :] = params[j * self.d : (j + 1) * self.d]
        return A

    def _learn_lyapunov(self, epsilon: float = 1e-2) -> np.ndarray:
        """
        Learn P using convex optimization with scipy.
        Parameterize P as symmetric matrix with vector of 6 params for 3D.
        Assume D=3.
        """
        if self.d != 3:
            raise ValueError("Lyapunov learning implemented for D=3 only.")
        delta_x = self.demo_positions - self.x_star
        dot_x = self.demo_velocities

        def objective(v: np.ndarray) -> float:
            P = np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])
            violations = np.maximum(0, np.einsum("ni,ij,nj->n", dot_x, P, delta_x))
            return np.sum(violations)

        def psd_constraint(v: np.ndarray) -> float:
            P = np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])
            return np.min(np.linalg.eigvalsh(P)) - epsilon

        initial_v = np.array([1, 0, 0, 1, 0, 1])  # Identity flattened
        constraints = {"type": "ineq", "fun": psd_constraint}
        res = minimize(
            objective,
            initial_v,
            method="SLSQP",
            constraints=[constraints],
            options={"maxiter": 200},
        )
        if res.success:
            v = res.x
            P = np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])
            return P
        else:
            print("Lyapunov optimization failed, using identity.")
            return np.eye(self.d)

    def adapt(self, new_x_star: np.ndarray) -> LPVDS:
        """
        Adapt the policy to new attractor using Laplacian editing on means.
        """
        mu = self.gmm_params.mu.copy()  # K x D
        K = self.k

        # Build Laplacian for chain graph
        L = np.zeros((K, K))
        for i in range(K):
            if i > 0:
                L[i, i - 1] = -1
            L[i, i] = 1 if i == 0 or i == K - 1 else 2
            if i < K - 1:
                L[i, i + 1] = -1

        Delta = L @ mu  # K x D

        # Constraints: fix first mean, set last to new_x_star
        fixed_indices = [0, K - 1]
        fixed_values = np.zeros((2, self.d))
        fixed_values[0] = mu[0]
        fixed_values[1] = new_x_star

        free_indices = list(range(1, K - 1))
        if not free_indices:
            raise ValueError("K too small for adaptation.")

        # For each dimension
        new_mu = mu.copy()
        for dim in range(self.d):
            A = (L.T @ L)[free_indices, :][:, free_indices]
            b = (L.T @ Delta[:, dim])[free_indices] - (L.T @ L)[free_indices, :][
                :, fixed_indices
            ] @ fixed_values[:, dim]
            new_mu_free = np.linalg.solve(A, b)
            new_mu[free_indices, dim] = new_mu_free

        # Update GMM params
        new_gmm_params = GMMParams(
            self.gmm_params.pi, new_mu, self.gmm_params.sigma.copy()
        )

        # Re-learn DS params and P
        new_A = (
            self._learn_ds_params()
        )  # Uses original data, but new gamma from new_gmm_params
        # Temporarily update gmm_params for re-learning
        original_gmm = self.gmm_params
        self.gmm_params = new_gmm_params
        new_A = self._learn_ds_params()
        new_P = self._learn_lyapunov()
        self.gmm_params = original_gmm  # Restore

        return LPVDS(new_gmm_params, new_A, new_x_star, new_P)


# Example usage (dummy data for illustration)
if __name__ == "__main__":
    N, D, K = 100, 3, 5
    t = np.linspace(0, 1, N)
    demo_positions = np.column_stack(
        (t, np.zeros(N), np.zeros(N))
    )  # Simple line trajectory
    demo_velocities = np.column_stack(
        (np.ones(N), np.zeros(N), np.zeros(N))
    )  # Constant velocity
    x_star = demo_positions[-1]

    emp = EMP(K, demo_positions, demo_velocities, x_star)
    print("Original policy ready.")

    new_x_star = x_star + np.array([0.5, 0.5, 0.5])
    adapted_policy = emp.adapt(new_x_star)
    print("Adapted policy ready.")

    test_x = np.array([0.5, 0, 0])
    vel = adapted_policy.compute_velocity(test_x)
    print(f"Velocity at {test_x}: {vel}")
