from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
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
        n, d = data.shape
        # Initialize
        pi = np.ones(self.k) / self.k
        idx = np.random.choice(n, self.k, replace=False)
        mu = data[idx]
        sigma = np.array([np.eye(d) for _ in range(self.k)])
        prev_loglik = -np.inf

        for it in range(max_iter):
            # E-step
            probs = np.zeros((n, self.k))
            for j in range(self.k):
                probs[:, j] = pi[j] * multivariate_normal.pdf(
                    data, mean=mu[j], cov=sigma[j], allow_singular=True
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

            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik

        return GMMParams(pi, mu, sigma)

    def get_gamma(self, params: GMMParams, x: np.ndarray) -> np.ndarray:
        probs = np.zeros(params.pi.shape[0])
        for j in range(len(probs)):
            probs[j] = params.pi[j] * multivariate_normal.pdf(
                x, params.mu[j], params.sigma[j], allow_singular=True
            )
        probs /= probs.sum() + 1e-10
        return probs


# Quaternion helper functions
def q_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def q_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y1
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x1
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def q_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def q_log(
    q: np.ndarray, ref: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
) -> np.ndarray:
    q = q_normalize(q)
    ref = q_normalize(ref)
    q_rel = q_mult(q_conj(ref), q)
    w = np.clip(q_rel[0], -1, 1)
    if np.abs(w) >= 1 - 1e-6:
        return np.zeros(3)
    theta = 2 * np.arccos(w)
    v = q_rel[1:]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        return np.zeros(3)
    return theta * (v / norm_v)


def q_exp(
    v: np.ndarray, ref: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
) -> np.ndarray:
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return q_normalize(ref)
    u = v / theta
    half_theta = theta / 2
    q_delta = np.array([np.cos(half_theta), *(np.sin(half_theta) * u)])
    return q_normalize(q_mult(ref, q_delta))


class LPVDSPos:
    def __init__(
        self,
        gmm_params: GMMParams,
        A: np.ndarray,
        x_star: np.ndarray,
        P: np.ndarray | None = None,
    ):
        self.gmm_params = gmm_params
        self.A = A  # (K, D, D)
        self.x_star = x_star
        self.P = P
        self.gmm = GMM(gmm_params.pi.size)

    def compute_velocity(self, x: np.ndarray) -> np.ndarray:
        gamma = self.gmm.get_gamma(self.gmm_params, x)
        dot_x = np.zeros_like(x)
        for j in range(gamma.size):
            dot_x += gamma[j] * (self.A[j] @ (x - self.x_star))
        return dot_x


class LPVDSOri:
    def __init__(
        self,
        gmm_params: GMMParams,
        A: np.ndarray,
        q_star: np.ndarray,
        P: np.ndarray | None = None,
    ):
        self.gmm_params = gmm_params
        self.A = A  # (K, 3, 3)
        self.q_star = q_normalize(q_star)
        self.P = P
        self.gmm = GMM(gmm_params.pi.size)

    def compute_omega(self, q: np.ndarray) -> np.ndarray:
        q = q_normalize(q)
        gamma = self.gmm.get_gamma(self.gmm_params, q)
        log_q = q_log(q, self.q_star)
        omega = np.zeros(3)
        for j in range(gamma.size):
            omega += gamma[j] * (self.A[j] @ log_q)
        return omega

    def compute_velocity(self, q: np.ndarray) -> np.ndarray:
        omega = self.compute_omega(q)
        q_omega = np.concatenate(([0.0], omega))
        dot_q = 0.5 * q_mult(q_omega, q_normalize(q))
        return dot_q


class EMPPos:
    def __init__(
        self,
        k: int,
        demo_positions: np.ndarray,
        demo_velocities: np.ndarray,
        x_star: np.ndarray,
    ):
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
        self.policy = LPVDSPos(self.gmm_params, self.A, x_star, self.P)

    def _order_components(self) -> GMMParams:
        n = self.demo_positions.shape[0]
        times = np.arange(n) / (n - 1.0)
        probs = np.zeros((n, self.k))
        for j in range(self.k):
            probs[:, j] = self.gmm_params.pi[j] * multivariate_normal.pdf(
                self.demo_positions,
                self.gmm_params.mu[j],
                self.gmm_params.sigma[j],
                allow_singular=True,
            )
        probs /= probs.sum(1, keepdims=True) + 1e-10
        weighted_times = np.dot(probs.T, times) / (probs.sum(0) + 1e-10)
        order = np.argsort(weighted_times)
        return GMMParams(
            self.gmm_params.pi[order],
            self.gmm_params.mu[order],
            self.gmm_params.sigma[order],
        )

    def _learn_ds_params(self) -> np.ndarray:
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
        if self.d != 3 and self.d != 2:
            print("Lyapunov for D=2 or 3 only, using identity.")
            return np.eye(self.d)
        delta_x = self.demo_positions - self.x_star
        dot_x = self.demo_velocities

        def make_P(v):
            if self.d == 2:
                return np.array([[v[0], v[1]], [v[1], v[2]]])
            else:
                return np.array(
                    [[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]]
                )

        def objective(v: np.ndarray) -> float:
            P = make_P(v)
            dV = np.einsum("ni,ij,nj->n", dot_x, P, delta_x)
            violations = np.maximum(0, dV)
            return np.sum(violations)

        def psd_constraint(v: np.ndarray) -> float:
            P = make_P(v)
            return np.min(np.linalg.eigvalsh(P)) - epsilon

        if self.d == 2:
            initial_v = np.array([1, 0, 1])
        else:
            initial_v = np.array([1, 0, 0, 1, 0, 1])
        constraints = {"type": "ineq", "fun": psd_constraint}
        res = minimize(
            objective,
            initial_v,
            method="SLSQP",
            constraints=[constraints],
            options={"maxiter": 500},
        )
        if res.success:
            return make_P(res.x)
        else:
            print("Lyapunov optimization failed, using identity.")
            return np.eye(self.d)

    def adapt(self, new_x_star: np.ndarray, epsilon: float = 1e-2) -> LPVDSPos:
        mu = self.gmm_params.mu.copy()
        K = self.k

        L = np.zeros((K, K))
        for i in range(K):
            L[i, i] = 1 if i in (0, K - 1) else 2
            if i > 0:
                L[i, i - 1] = -1
            if i < K - 1:
                L[i, i + 1] = -1

        Delta = L @ mu

        fixed_indices = [0, K - 1]
        fixed_values = np.zeros((2, self.d))
        fixed_values[0] = mu[0]
        fixed_values[1] = new_x_star
        free_indices = list(range(1, K - 1))
        if not free_indices:
            raise ValueError("K too small.")

        new_mu = mu.copy()
        for dim in range(self.d):
            A = L.T @ L
            A_free = A[np.ix_(free_indices, free_indices)]
            b = (L.T @ Delta[:, dim])[free_indices] - A[
                np.ix_(free_indices, fixed_indices)
            ] @ fixed_values[:, dim]
            new_mu_free = np.linalg.solve(A_free, b)
            new_mu[np.array(free_indices), dim] = new_mu_free

        new_gmm_params = GMMParams(
            self.gmm_params.pi, new_mu, self.gmm_params.sigma.copy()
        )

        # Re-learn A and P with new gmm and new_x_star
        new_delta_x = self.demo_positions - new_x_star
        n = new_delta_x.shape[0]
        new_A = np.zeros((self.k, self.d, self.d))
        for out_dim in range(self.d):
            y = self.demo_velocities[:, out_dim]
            X = np.zeros((n, self.k * self.d))
            for i in range(n):
                gamma = self.gmm.get_gamma(new_gmm_params, self.demo_positions[i])
                for j in range(self.k):
                    X[i, j * self.d : (j + 1) * self.d] = gamma[j] * new_delta_x[i]
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            for j in range(self.k):
                new_A[j, out_dim, :] = params[j * self.d : (j + 1) * self.d]

        # Re-learn P
        def make_P(v):
            if self.d == 2:
                return np.array([[v[0], v[1]], [v[1], v[2]]])
            else:
                return np.array(
                    [[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]]
                )

        def objective(v: np.ndarray) -> float:
            P = make_P(v)
            dV = np.einsum("ni,ij,nj->n", self.demo_velocities, P, new_delta_x)
            violations = np.maximum(0, dV)
            return np.sum(violations)

        def psd_constraint(v: np.ndarray) -> float:
            P = make_P(v)
            return np.min(np.linalg.eigvalsh(P)) - epsilon

        if self.d == 2:
            initial_v = np.array([1, 0, 1])
        else:
            initial_v = np.array([1, 0, 0, 1, 0, 1])
        res = minimize(
            objective,
            initial_v,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": psd_constraint}],
            options={"maxiter": 500},
        )
        if res.success:
            new_P = make_P(res.x)
        else:
            new_P = np.eye(self.d)

        return LPVDSPos(new_gmm_params, new_A, new_x_star, new_P)


class EMPOri:
    def __init__(
        self,
        k: int,
        demo_quats: np.ndarray,
        demo_omegas: np.ndarray,
        q_star: np.ndarray,
    ):
        self.k = k
        self.demo_quats = q_normalize(demo_quats)
        self.demo_omegas = demo_omegas
        self.q_star = q_normalize(q_star)
        self.d = 3  # Tangent space dim
        self.gmm = GMM(k)
        self.gmm_params = self.gmm.fit(self.demo_quats)
        self.gmm_params = self._order_components()
        self.A = self._learn_ds_params()
        self.P = self._learn_lyapunov()
        self.policy = LPVDSOri(self.gmm_params, self.A, q_star, self.P)

    def _order_components(self) -> GMMParams:
        n = self.demo_quats.shape[0]
        times = np.arange(n) / (n - 1.0)
        probs = np.zeros((n, self.k))
        for j in range(self.k):
            probs[:, j] = self.gmm_params.pi[j] * multivariate_normal.pdf(
                self.demo_quats,
                self.gmm_params.mu[j],
                self.gmm_params.sigma[j],
                allow_singular=True,
            )
        probs /= probs.sum(1, keepdims=True) + 1e-10
        weighted_times = np.dot(probs.T, times) / (probs.sum(0) + 1e-10)
        order = np.argsort(weighted_times)
        return GMMParams(
            self.gmm_params.pi[order],
            self.gmm_params.mu[order],
            self.gmm_params.sigma[order],
        )

    def _learn_ds_params(self) -> np.ndarray:
        n = self.demo_quats.shape[0]
        A = np.zeros((self.k, self.d, self.d))
        logs = np.array([q_log(self.demo_quats[i], self.q_star) for i in range(n)])
        for out_dim in range(self.d):
            y = self.demo_omegas[:, out_dim]
            X = np.zeros((n, self.k * self.d))
            for i in range(n):
                gamma = self.gmm.get_gamma(self.gmm_params, self.demo_quats[i])
                for j in range(self.k):
                    X[i, j * self.d : (j + 1) * self.d] = gamma[j] * logs[i]
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            for j in range(self.k):
                A[j, out_dim, :] = params[j * self.d : (j + 1) * self.d]
        return A

    def _learn_lyapunov(self, epsilon: float = 1e-2) -> np.ndarray:
        logs = np.array([q_log(q, self.q_star) for q in self.demo_quats])
        omegas = self.demo_omegas

        def make_P(v):
            return np.array(
                [[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]]
            )

        def objective(v: np.ndarray) -> float:
            P = make_P(v)
            dV = 2 * np.einsum("ni,ij,nj->n", omegas, P, logs)
            violations = np.maximum(0, dV)
            return np.sum(violations)

        def psd_constraint(v: np.ndarray) -> float:
            P = make_P(v)
            return np.min(np.linalg.eigvalsh(P)) - epsilon

        initial_v = np.array([1, 0, 0, 1, 0, 1])
        constraints = {"type": "ineq", "fun": psd_constraint}
        res = minimize(
            objective,
            initial_v,
            method="SLSQP",
            constraints=[constraints],
            options={"maxiter": 500},
        )
        if res.success:
            return make_P(res.x)
        else:
            print("Lyapunov optimization failed, using identity.")
            return np.eye(self.d)

    def adapt(self, new_q_star: np.ndarray, epsilon: float = 1e-2) -> LPVDSOri:
        new_q_star = q_normalize(new_q_star)
        mu = self.gmm_params.mu.copy()
        K = self.k
        mu0 = q_normalize(mu[0])

        logs = np.array([q_log(mu[j], mu0) for j in range(K)])

        L = np.zeros((K, K))
        for i in range(K):
            L[i, i] = 1 if i in (0, K - 1) else 2
            if i > 0:
                L[i, i - 1] = -1
            if i < K - 1:
                L[i, i + 1] = -1

        Delta = L @ logs

        fixed_indices = [0, K - 1]
        fixed_values = np.zeros((2, self.d))
        fixed_values[0] = logs[0]
        fixed_values[1] = q_log(new_q_star, mu0)
        free_indices = list(range(1, K - 1))
        if not free_indices:
            raise ValueError("K too small.")

        new_logs = logs.copy()
        for dim in range(self.d):
            A = L.T @ L
            A_free = A[np.ix_(free_indices, free_indices)]
            b = (L.T @ Delta[:, dim])[free_indices] - A[
                np.ix_(free_indices, fixed_indices)
            ] @ fixed_values[:, dim]
            new_logs_free = np.linalg.solve(A_free, b)
            new_logs[np.array(free_indices), dim] = new_logs_free

        new_mu = np.array([q_exp(new_logs[j], mu0) for j in range(K)])

        new_gmm_params = GMMParams(
            self.gmm_params.pi, new_mu, self.gmm_params.sigma.copy()
        )

        # Re-learn A and P with new gmm and new_q_star
        n = self.demo_quats.shape[0]
        new_A = np.zeros((self.k, self.d, self.d))
        new_logs = np.array([q_log(self.demo_quats[i], new_q_star) for i in range(n)])
        for out_dim in range(self.d):
            y = self.demo_omegas[:, out_dim]
            X = np.zeros((n, self.k * self.d))
            for i in range(n):
                gamma = self.gmm.get_gamma(new_gmm_params, self.demo_quats[i])
                for j in range(self.k):
                    X[i, j * self.d : (j + 1) * self.d] = gamma[j] * new_logs[i]
            params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            for j in range(self.k):
                new_A[j, out_dim, :] = params[j * self.d : (j + 1) * self.d]

        # Re-learn P
        def make_P(v):
            return np.array(
                [[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]]
            )

        def objective(v: np.ndarray) -> float:
            P = make_P(v)
            dV = 2 * np.einsum("ni,ij,nj->n", self.demo_omegas, P, new_logs)
            violations = np.maximum(0, dV)
            return np.sum(violations)

        def psd_constraint(v: np.ndarray) -> float:
            P = make_P(v)
            return np.min(np.linalg.eigvalsh(P)) - epsilon

        initial_v = np.array([1, 0, 0, 1, 0, 1])
        res = minimize(
            objective,
            initial_v,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": psd_constraint}],
            options={"maxiter": 500},
        )
        if res.success:
            new_P = make_P(res.x)
        else:
            new_P = np.eye(self.d)

        return LPVDSOri(new_gmm_params, new_A, new_q_star, new_P)


# To replicate figures, e.g., Fig. 2 in the paper (S-shaped 2D trajectory)
# Use this example code to generate and plot. Run it locally with matplotlib.

if __name__ == "__main__":
    # Example for positions (2D S-shape like in paper)
    N, D, K = 200, 2, 3
    t = np.linspace(0, 1, N)
    demo_positions = np.column_stack((t, 0.2 * np.sin(4 * np.pi * t)))
    dt = t[1] - t[0]
    demo_velocities = (
        np.column_stack((np.ones(N), 0.2 * 4 * np.pi * np.cos(4 * np.pi * t))) * dt
    )  # Scaled for small steps
    x_star = demo_positions[-1]

    emp_pos = EMPPos(K, demo_positions, demo_velocities, x_star)
    print("Positional policy ready.")

    # Simulate trajectory from start
    def ode_func(x, t, policy):
        return policy.compute_velocity(x)

    sim_t = np.linspace(0, 1, N)
    sim_pos = odeint(ode_func, demo_positions[0], sim_t, args=(emp_pos.policy,))

    # Plot demonstration vs learned
    plt.figure()
    plt.plot(demo_positions[:, 0], demo_positions[:, 1], "b-", label="Demo")
    plt.plot(sim_pos[:, 0], sim_pos[:, 1], "r--", label="Learned")
    plt.scatter(x_star[0], x_star[1], c="g", marker="*", label="Attractor")
    plt.legend()
    plt.title("Learned DS Trajectory")
    plt.show()

    # Streamplot for 2D
    xx, yy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(-0.3, 0.3, 20))
    uu, vv = np.zeros_like(xx), np.zeros_like(yy)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            vel = emp_pos.policy.compute_velocity(np.array([xx[i, j], yy[i, j]]))
            uu[i, j] = vel[0]
            vv[i, j] = vel[1]
    plt.figure()
    plt.streamplot(xx, yy, uu, vv, color="b")
    plt.plot(demo_positions[:, 0], demo_positions[:, 1], "r-")
    plt.scatter(x_star[0], x_star[1], c="g", marker="*")
    plt.title("Streamlines of Learned DS")
    plt.show()

    # Adaptation
    new_x_star = x_star + np.array([0.0, 0.3])
    adapted_pos = emp_pos.adapt(new_x_star)
    sim_pos_adapted = odeint(ode_func, demo_positions[0], sim_t, args=(adapted_pos,))

    plt.figure()
    plt.plot(demo_positions[:, 0], demo_positions[:, 1], "b-", label="Original Demo")
    plt.plot(sim_pos_adapted[:, 0], sim_pos_adapted[:, 1], "g--", label="Adapted")
    plt.scatter(new_x_star[0], new_x_star[1], c="m", marker="*", label="New Attractor")
    plt.legend()
    plt.title("Adapted Trajectory")
    plt.show()

    # Example for orientations (rotation around z-axis)
    N, K = 100, 3
    t = np.linspace(0, 1, N)
    theta = np.pi * t
    demo_quats = np.column_stack(
        (np.cos(theta / 2), np.zeros(N), np.zeros(N), np.sin(theta / 2))
    )
    demo_omegas = np.tile(np.array([0, 0, np.pi]), (N, 1))  # Constant omega
    q_star = demo_quats[-1]

    emp_ori = EMPOri(K, demo_quats, demo_omegas, q_star)
    print("Orientation policy ready.")

    # Simulate quaternion trajectory
    def ode_func_q(q, t, policy):
        return policy.compute_velocity(q)

    sim_q = odeint(ode_func_q, demo_quats[0], sim_t, args=(emp_ori.policy,))

    # To visualize orientations, one way is to apply to basis vectors, but for simplicity, plot angles
    angles = 2 * np.arccos(sim_q[:, 0])
    plt.figure()
    plt.plot(sim_t, angles, "r--", label="Learned Angle")
    plt.plot(t, theta, "b-", label="Demo Angle")
    plt.legend()
    plt.title("Learned Orientation Trajectory (Angle)")
    plt.show()

    # Adaptation to larger angle
    new_theta = 1.5 * np.pi
    new_q_star = np.array([np.cos(new_theta / 2), 0, 0, np.sin(new_theta / 2)])
    adapted_ori = emp_ori.adapt(new_q_star)
    sim_q_adapted = odeint(ode_func_q, demo_quats[0], sim_t, args=(adapted_ori,))
    new_angles = 2 * np.arccos(sim_q_adapted[:, 0])
    plt.figure()
    plt.plot(t, theta, "b-", label="Original Demo")
    plt.plot(sim_t, new_angles, "g--", label="Adapted")
    plt.legend()
    plt.title("Adapted Orientation Trajectory (Angle)")
    plt.show()
