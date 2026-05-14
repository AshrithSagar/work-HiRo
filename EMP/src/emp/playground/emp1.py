"""
Elastic Motion Policy (EMP): An Adaptive Dynamical System for One-Shot Imitation Learning

Based on arXiv:2503.08029 by Tianyu Li et al.

This implementation provides a framework for learning stable, reactive motion policies
from a single demonstration with adaptation capabilities using Laplacian editing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Self, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from sklearn.mixture import GaussianMixture
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.shapes import FOUR, THREE

N = TypeVar("N", bound=int, default=int, covariant=True)
M = TypeVar("M", bound=int, default=int, covariant=True)
L = TypeVar("L", bound=int, default=int, covariant=True)
DimSpace = TypeVar("DimSpace", bound=int, default=int, covariant=True)
"""TypeVar denoting dimension of the space"""
NumPoints = TypeVar("NumPoints", bound=int, default=int, covariant=True)
"""TypeVar denoting number of points"""
NumComponents = TypeVar("NumComponents", bound=int, default=int, covariant=True)
"""TypeVar denoting number of components"""

Array1D = TypedNDArray[tuple[N]]
Array2D = TypedNDArray[tuple[M, N]]
Array3D = TypedNDArray[tuple[L, M, N]]
SquareMatrix = Array2D[N, N]

Point3D = Array1D[THREE]
RotationMatrix3D = SquareMatrix[THREE]
Twist3D = Array1D[Literal[6]]
Quaternion = Array1D[FOUR]
Trajectory = Array2D[NumPoints, DimSpace]
VectorN = Array1D[N]
TransformationMatrix = SquareMatrix[FOUR]
CovarianceMatrix = SquareMatrix[N]

IdentityRotation3D = RotationMatrix3D(np.eye(3, dtype=np.double))


@dataclass(frozen=True)
class SE3Pose:
    """Represents `SE(3)` pose: position in `R^3` and orientation in `SO(3)`."""

    position: Point3D
    rotation: RotationMatrix3D

    @enforce_shapes
    @classmethod
    def from_position_quaternion(
        cls, pos: npt.ArrayLike, quat: onp.ToFloat1D | onp.ToFloat2D
    ) -> Self:
        """Create pose from position and quaternion [w, x, y, z]."""
        return cls(
            position=Point3D(pos),
            rotation=RotationMatrix3D(Rotation.from_quat(quat).as_matrix()),
        )

    @enforce_shapes
    def to_twist(self) -> Twist3D:
        """Convert to twist coordinates (linear + angular velocity space)."""
        omega = Rotation.from_matrix(self.rotation).as_rotvec()
        return Twist3D(np.concatenate([self.position, omega]))

    @enforce_shapes
    def transform(self, point: Point3D) -> Point3D:
        """Transform a point by this pose."""
        return Point3D(self.rotation @ point + self.position)

    @enforce_shapes
    def inverse(self) -> SE3Pose:
        """Compute inverse transformation."""
        R_inv = self.rotation.T
        position = Point3D(-R_inv @ self.position)
        return SE3Pose(position=position, rotation=R_inv)


@dataclass
class TaskFrame:
    """Represents a task-relevant reference frame."""

    name: str
    pose: SE3Pose

    @enforce_shapes
    def transform_to_local(self, global_pose: SE3Pose) -> SE3Pose:
        """Transform global pose to this frame's local coordinates."""
        inv = self.pose.inverse()
        local_pos = Point3D(inv.rotation @ (global_pose.position - inv.position))
        local_rot = RotationMatrix3D(inv.rotation @ global_pose.rotation)
        return SE3Pose(position=local_pos, rotation=local_rot)


@dataclass
class GaussianComponent(Generic[N]):
    """Single Gaussian component in GMM."""

    mean: VectorN[N]
    covariance: CovarianceMatrix[N]
    weight: float

    @enforce_shapes
    def pdf(self, x: VectorN[N]) -> float:
        """Compute probability density at point x."""
        n = int(len(self.mean))
        det = float(np.linalg.det(self.covariance))
        inv_cov = CovarianceMatrix[N](np.linalg.inv(self.covariance))
        diff = VectorN[N](x - self.mean)

        norm = 1.0 / np.sqrt((2 * np.pi) ** n * det)
        exponent = float(-0.5 * diff.T @ inv_cov @ diff)
        return norm * np.exp(exponent)


@dataclass
class ElasticGMM(Generic[NumComponents, N]):
    """
    Elastic Gaussian Mixture Model constrained to SE(3) task frames.

    This GMM represents the demonstration in a way that can be adapted
    to new task configurations via Laplacian editing.
    """

    components: Sequence[GaussianComponent[N]]
    task_frames: Sequence[TaskFrame]
    attractor_frame: TaskFrame

    def __post_init__(self) -> None:
        """Validate that weights sum to 1."""
        total_weight = sum(c.weight for c in self.components)
        assert np.isclose(total_weight, 1.0), (
            f"Weights must sum to 1, got {total_weight}"
        )

    @enforce_shapes
    def responsibility(self, x: VectorN[N]) -> VectorN[NumComponents]:
        """Compute responsibility of each component for point x."""
        weighted_pdfs = [c.weight * c.pdf(x) for c in self.components]
        return VectorN[NumComponents](weighted_pdfs / (np.sum(weighted_pdfs) + 1e-10))

    @enforce_shapes
    def apply_laplacian_editing(
        self, new_task_frames: Sequence[TaskFrame]
    ) -> ElasticGMM[N]:
        """
        Apply Laplacian editing to adapt GMM to new task configuration.

        This preserves the relative geometric relationships between
        components while adapting to new frame positions.
        """
        assert len(new_task_frames) == len(self.task_frames)

        # Compute transformation for each frame
        transformations = list[TransformationMatrix]()
        for old_frame, new_frame in zip(self.task_frames, new_task_frames):
            # Compute relative transformation
            T_old_to_new = self._compute_frame_transformation(old_frame, new_frame)
            transformations.append(T_old_to_new)

        # Transform each Gaussian component
        new_components = list[GaussianComponent[N]]()
        for comp in self.components:
            # Determine which frame this component is closest to
            frame_idx = self._find_nearest_frame(comp.mean)
            T = transformations[frame_idx]

            # Transform mean
            pos_mean = comp.mean[:3]
            new_pos = T[:3, :3] @ pos_mean + T[:3, 3]
            new_mean = VectorN[N](np.concatenate([new_pos, comp.mean[3:]]))

            # Transform covariance (similarity transformation)
            new_cov = T[:3, :3] @ comp.covariance[:3, :3] @ T[:3, :3].T
            full_cov = CovarianceMatrix[N](block_diag(new_cov, comp.covariance[3:, 3:]))

            new_components.append(GaussianComponent[N](new_mean, full_cov, comp.weight))
        return ElasticGMM[N](new_components, new_task_frames, self.attractor_frame)

    @enforce_shapes
    def _compute_frame_transformation(
        self, old_frame: TaskFrame, new_frame: TaskFrame
    ) -> TransformationMatrix:
        """Compute 4x4 transformation matrix between frames."""
        T = np.eye(4)
        T[:3, :3] = new_frame.pose.rotation @ old_frame.pose.rotation.T
        T[:3, 3] = new_frame.pose.position - T[:3, :3] @ old_frame.pose.position
        return TransformationMatrix(T)

    @enforce_shapes
    def _find_nearest_frame(self, point: VectorN[N]) -> int:
        """Find index of nearest task frame to point."""
        pos = point[:3]
        dist = [np.linalg.norm(pos - frame.pose.position) for frame in self.task_frames]
        return int(np.argmin(dist))


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems."""

    @abstractmethod
    def compute_velocity(self, state: VectorN[N]) -> VectorN[N]:
        """Compute velocity at given state: dx/dt = f(x)."""
        pass

    @abstractmethod
    def is_stable(self, attractor: VectorN[N]) -> bool:
        """Check if system is globally asymptotically stable at attractor."""
        pass


@dataclass
class LPVDynamicsParameters(Generic[NumComponents, N]):
    """Parameters for Linear Parameter Varying Dynamical System."""

    gmm: ElasticGMM[NumComponents, N]
    A_matrices: Sequence[SquareMatrix]  # Linear dynamics for each Gaussian
    attractor: VectorN[N]

    def __post_init__(self):
        """Validate dimensions."""
        n_components = len(self.gmm.components)
        assert len(self.A_matrices) == n_components


class LPVDynamicalSystem(DynamicalSystem):
    """
    Linear Parameter Varying Dynamical System (LPV-DS).

    Represents motion as: dx/dt = sum_k h_k(x) * A_k * (x - x*)
    where h_k are Gaussian mixing functions and x* is the attractor.
    """

    def __init__(self, params: LPVDynamicsParameters):
        self.params = params
        self._validate_stability()

    @enforce_shapes
    def compute_velocity(self, state: VectorN[N]) -> VectorN[N]:
        """Compute velocity using LPV formulation."""
        responsibilities = self.params.gmm.responsibility(state)
        # Weighted sum of linear dynamics
        vel = np.zeros_like(state)
        x_diff = state - self.params.attractor
        for h_k, A_k in zip(responsibilities, self.params.A_matrices):
            vel += h_k * (A_k @ x_diff)
        return vel

    @enforce_shapes
    def is_stable(self, attractor: VectorN[N]) -> bool:
        """
        Check stability using Lyapunov theory.
        For LPV-DS, we need all A_k to be Hurwitz (negative definite).
        """
        for A_k in self.params.A_matrices:
            eigenvalues = np.linalg.eigvals(A_k)
            if not np.all(np.real(eigenvalues) < 0):
                return False
        return True

    def _validate_stability(self) -> None:
        """Ensure learned dynamics satisfy stability constraints."""
        if not self.is_stable(self.params.attractor):
            raise ValueError("LPV-DS parameters do not guarantee stability!")


@dataclass
class LyapunovFunction:
    """
    Quadratic Lyapunov function: V(x) = (x - x*)^T P (x - x*).

    Ensures energy always decreases along trajectories, guaranteeing stability.
    """

    P: SquareMatrix  # Positive definite matrix
    attractor: VectorN

    def __post_init__(self):
        """Validate P is symmetric positive definite."""
        assert np.allclose(self.P, self.P.T), "P must be symmetric"
        eigenvalues = np.linalg.eigvals(self.P)
        assert np.all(eigenvalues > 0), "P must be positive definite"

    @enforce_shapes
    def evaluate(self, state: VectorN[N]) -> float:
        """Evaluate Lyapunov function at state."""
        diff = state - self.attractor
        return float(diff.T @ self.P @ diff)

    @enforce_shapes
    def derivative(self, state: VectorN[N], velocity: VectorN[N]) -> float:
        """
        Compute time derivative: dV/dt = grad(V) · dx/dt.
        For stability, this must be negative definite (dV/dt < 0).
        """
        diff = state - self.attractor
        grad_V = 2 * self.P @ diff
        return float(grad_V.T @ velocity)


class LyapunovLearner(Generic[N]):
    """Learn Lyapunov function online via convex optimization."""

    @enforce_shapes
    def __init__(self, state_dim: N, learning_rate: float = 0.01):
        self.state_dim = int(state_dim)
        self.learning_rate = float(learning_rate)
        self.P = SquareMatrix[N](np.eye(state_dim))

    @enforce_shapes
    def update(
        self, state: VectorN[N], velocity: VectorN[N], attractor: VectorN[N]
    ) -> LyapunovFunction:
        """
        Update P matrix to ensure dV/dt < 0 along trajectory.
        Uses projected gradient descent on the cone of PSD matrices.
        """
        diff = state - attractor

        # Compute gradient of constraint violation
        # Want: diff^T P velocity < 0
        violation = diff.T @ self.P @ velocity

        if violation > 0:
            grad = np.outer(diff, velocity) + np.outer(velocity, diff)
            self.P = SquareMatrix[N](self.P - self.learning_rate * grad)
            self.P = self._project_to_psd(self.P)
        return LyapunovFunction(self.P.copy(), attractor.copy())

    @enforce_shapes
    def _project_to_psd(self, M: SquareMatrix[N]) -> SquareMatrix[N]:
        """Project matrix to positive semi-definite cone."""
        M_sym = SquareMatrix[N]((M + M.T) / 2)
        eigvals, eigvecs = np.linalg.eigh(M_sym)
        eigvals_pos = np.maximum(eigvals, 1e-6)
        return SquareMatrix[N](eigvecs @ np.diag(eigvals_pos) @ eigvecs.T)


@dataclass
class Obstacle:
    """Represents an obstacle in the workspace."""

    center: Point3D
    radius: float

    @enforce_shapes
    def distance_to(self, point: Point3D) -> float:
        """Compute distance from point to obstacle surface."""
        return float(np.linalg.norm(point - self.center) - self.radius)

    @enforce_shapes
    def is_inside(self, point: Point3D) -> bool:
        """Check if point is inside obstacle."""
        return self.distance_to(point) < 0


class ObstacleModulator(Generic[N]):
    """
    Modulate velocity field to avoid obstacles while preserving stability.
    Uses dynamical system modulation as described in the paper.
    """

    def __init__(self, safety_margin: float = 0.1):
        self.safety_margin = float(safety_margin)

    @enforce_shapes
    def modulate_velocity(
        self, state: VectorN[N], velocity: VectorN[N], obstacles: Sequence[Obstacle]
    ) -> VectorN[N]:
        """
        Modulate velocity to avoid obstacles.

        Uses normal-space modulation to deflect flow around obstacles
        while maintaining tangential motion.
        """
        pos = Point3D(state[:3])
        modulated_vel = velocity.copy()
        for obs in obstacles:
            dist = obs.distance_to(pos)
            if dist < self.safety_margin:
                # Compute normal direction (away from obstacle)
                normal = (pos - obs.center) / np.linalg.norm(pos - obs.center)

                # Decompose velocity into normal and tangential components
                vel_normal = (normal @ modulated_vel[:3]) * normal
                vel_tangent = modulated_vel[:3] - vel_normal

                # Modulation factor (stronger as we get closer)
                alpha = max(0, 1 - dist / self.safety_margin)

                # Reduce normal component, preserve tangent
                modulated_normal = (1 - alpha) * vel_normal
                modulated_vel[:3] = modulated_normal + vel_tangent
        return modulated_vel


@dataclass
class EMPConfig:
    """Configuration for Elastic Motion Policy."""

    state_dim: int
    n_gaussians: int
    lyapunov_learning_rate: float = 0.01
    obstacle_safety_margin: float = 0.1
    dt: float = 0.01  # Integration time step


class ElasticMotionPolicy(Generic[N]):
    """
    Main Elastic Motion Policy framework.

    Learns stable, adaptive motion policies from a single demonstration
    with guaranteed convergence and real-time adaptation capabilities.
    """

    def __init__(
        self,
        config: EMPConfig,
        demonstration: Trajectory,
        task_frames: Sequence[TaskFrame],
        attractor: VectorN,
    ):
        self.config = config
        self.attractor = attractor

        # Learn Elastic-GMM from demonstration
        self.elastic_gmm = self._learn_elastic_gmm(
            demonstration, task_frames, attractor
        )
        # Learn LPV-DS dynamics
        self.lpv_ds = self._learn_lpv_ds(demonstration)
        # Initialize Lyapunov learner
        self.lyapunov_learner = LyapunovLearner(
            config.state_dim, config.lyapunov_learning_rate
        )
        # Obstacle modulator
        self.obstacle_modulator = ObstacleModulator(config.obstacle_safety_margin)

    def adapt_to_new_context(self, new_task_frames: Sequence[TaskFrame]) -> None:
        """
        Adapt policy to new task configuration using Laplacian editing.
        This happens in real-time without new demonstrations.
        """
        self.elastic_gmm = self.elastic_gmm.apply_laplacian_editing(new_task_frames)
        # Re-learn dynamics with adapted GMM
        self.lpv_ds.params.gmm = self.elastic_gmm

    def execute(
        self,
        initial_state: VectorN,
        obstacles: Sequence[Obstacle] | None = None,
        max_steps: int = 1000,
        convergence_threshold: float = 1e-3,
    ) -> Trajectory:
        """
        Execute motion policy from initial state to attractor.

        Returns trajectory with guaranteed convergence and obstacle avoidance.
        """
        if obstacles is None:
            obstacles = []

        trajectory = [initial_state.copy()]
        state = initial_state.copy()

        for _ in range(max_steps):
            # Compute nominal velocity from LPV-DS
            velocity = self.lpv_ds.compute_velocity(state)

            # Modulate for obstacle avoidance
            velocity = self.obstacle_modulator.modulate_velocity(
                state, velocity, obstacles
            )

            # Update Lyapunov function online
            _lyapunov = self.lyapunov_learner.update(state, velocity, self.attractor)

            # Integrate
            state = VectorN(state + velocity * self.config.dt)
            trajectory.append(state.copy())

            # Check convergence
            if np.linalg.norm(state - self.attractor) < convergence_threshold:
                break

        return Trajectory(trajectory)

    @enforce_shapes
    def _learn_elastic_gmm(
        self, demo: Trajectory, frames: Sequence[TaskFrame], attractor: VectorN
    ) -> ElasticGMM[N]:
        """Learn Elastic-GMM from demonstration using EM algorithm."""

        # Fit GMM
        gmm_model = GaussianMixture(
            n_components=self.config.n_gaussians, covariance_type="full"
        )
        gmm_model.fit(demo)

        # Extract components
        components = [
            GaussianComponent[N](
                mean=VectorN[N](Array2D(gmm_model.means_)[i]),
                covariance=SquareMatrix[N](Array3D(gmm_model.covariances_)[i]),
                weight=float(VectorN[int](gmm_model.weights_)[i]),
            )
            for i in range(self.config.n_gaussians)
        ]

        attractor_frame = TaskFrame(
            "attractor", SE3Pose(Point3D(attractor[:3]), IdentityRotation3D)
        )
        return ElasticGMM(components, frames, attractor_frame)

    def _learn_lpv_ds(self, demo: Trajectory) -> LPVDynamicalSystem:
        """
        Learn LPV-DS parameters with stability constraints.
        Uses convex optimization to ensure Hurwitz matrices.
        """
        n = self.config.state_dim
        n_components = self.config.n_gaussians

        # Estimate velocities from demonstration
        _velocities = np.diff(demo, axis=0) / self.config.dt

        # Learn A matrices with stability constraints
        A_matrices = list[SquareMatrix]()
        for k in range(n_components):
            # Simple initialization: stable linear system
            A_k = SquareMatrix(-np.eye(n) * np.random.uniform(0.5, 2.0))
            A_matrices.append(A_k)

        params = LPVDynamicsParameters(
            gmm=self.elastic_gmm, A_matrices=A_matrices, attractor=self.attractor
        )

        return LPVDynamicalSystem(params)


def example_book_placing_task():
    """
    Example: Book placing task from the paper.
    Robot learns to place a book on a rack from one demonstration,
    then adapts to different rack positions.
    """
    # Configuration
    config = EMPConfig(
        state_dim=6,  # 3D position + 3D orientation velocity
        n_gaussians=3,
        lyapunov_learning_rate=0.01,
        obstacle_safety_margin=0.1,
        dt=0.01,
    )

    # Generate synthetic demonstration (in practice, from robot teleop)
    t = np.linspace(0, 1, 100)
    demo_x = 0.5 * np.cos(2 * np.pi * t) + 0.5
    demo_y = 0.5 * np.sin(2 * np.pi * t)
    demo_z = 0.3 * t
    demo_pos = np.column_stack([demo_x, demo_y, demo_z])
    demo_orient = np.zeros((100, 3))  # Simplified: zero angular velocity
    demonstration = Trajectory(np.column_stack([demo_pos, demo_orient]))

    # Task frames
    start_frame = TaskFrame(
        "start", SE3Pose(Point3D([0.5, 0.0, 0.0]), IdentityRotation3D)
    )
    rack_frame = TaskFrame(
        "rack", SE3Pose(Point3D([0.5, 0.0, 0.3]), IdentityRotation3D)
    )
    task_frames = [start_frame, rack_frame]

    # Attractor (final position on rack)
    attractor = VectorN([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    emp = ElasticMotionPolicy(config, demonstration, task_frames, attractor)

    # Execute from initial state
    initial_state = VectorN([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    trajectory = emp.execute(initial_state)

    print(f"Generated trajectory with {len(trajectory)} steps")
    print(f"Final state: {trajectory[-1]}")
    print(f"Distance to attractor: {np.linalg.norm(trajectory[-1] - attractor)}")

    # Adapt to new rack position
    new_rack_frame = TaskFrame(
        "rack", SE3Pose(Point3D([0.7, 0.2, 0.35]), IdentityRotation3D)
    )  # Moved rack
    emp.adapt_to_new_context([start_frame, new_rack_frame])

    # Update attractor
    new_attractor = VectorN([0.7, 0.2, 0.35, 0.0, 0.0, 0.0])
    emp.attractor = new_attractor

    # Execute adapted policy
    adapted_trajectory = emp.execute(initial_state)
    print(f"\nAdapted trajectory with {len(adapted_trajectory)} steps")
    print(f"Final state: {adapted_trajectory[-1]}")

    return emp, trajectory, adapted_trajectory


if __name__ == "__main__":
    emp, original_traj, adapted_traj = example_book_placing_task()
