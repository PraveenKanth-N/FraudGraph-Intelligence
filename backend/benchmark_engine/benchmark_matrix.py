"""
Benchmark Matrix & Workload Configuration Specification — FraudGraph Intelligence (FGI)
Phase 8: Benchmark Automation & Experiment Management (FGI-MOD-08)

Defines the formal taxonomy and configurations for:
- Logical Workload Configurations (Graph topology, scale, feature dim, operator)
- Implementation-Specific Runs (Engine, execution boundary, block size, stream count)
- Measurement Repetitions and Warmup Iterations
- Execution Status Taxonomy
- Standard Profiles (smoke, scaling_features, scaling_graphs, power_law_vs_uniform, custom)
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ExecutionStatus(str, Enum):
    """Execution and verification status taxonomy for benchmark runs."""
    MEASURED = "MEASURED"
    NOT_RUN = "NOT_RUN"
    SKIPPED_CUDA_UNAVAILABLE = "SKIPPED_CUDA_UNAVAILABLE"
    SKIPPED_OOM_GUARD = "SKIPPED_OOM_GUARD"
    FAILED_TOLERANCE = "FAILED_TOLERANCE"
    FAILED_NON_FINITE = "FAILED_NON_FINITE"
    FAILED_SHAPE_MISMATCH = "FAILED_SHAPE_MISMATCH"
    FAILED_EXECUTION = "FAILED_EXECUTION"


# All supported aggregation engines
CPU_ENGINES = ["cpu_reference", "cpu_vectorized"]
CUDA_ENGINES = ["cuda_baseline", "cuda_coalesced", "cuda_streams", "cuda_degree_aware"]
ALL_ENGINES = CPU_ENGINES + CUDA_ENGINES

# Timing boundary mapping
TIMING_BOUNDARIES = {
    "cpu_reference": "cpu_host_timer",
    "cpu_vectorized": "cpu_host_timer",
    "cuda_baseline": "gpu_cuda_event",
    "cuda_coalesced": "gpu_cuda_event",
    "cuda_streams": "gpu_cuda_event",
    "cuda_degree_aware": "gpu_cuda_event",
}


@dataclass(frozen=True)
class WorkloadConfig:
    """
    Specification of a single logical workload configuration.

    A workload configuration defines the graph structure, scale, feature dimensionality,
    and aggregation operation independent of the underlying implementation engine.
    """
    name: str
    num_nodes: int
    num_edges: int
    feature_dim: int
    operator: str  # "mean", "sum", "max"
    is_directed: bool = True
    topology: str = "synthetic"  # "toy", "power_law", "uniform", "synthetic"
    graph_fixture: Optional[str] = None
    seed: Optional[int] = 42
    generator_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadConfig":
        return cls(**data)


@dataclass(frozen=True)
class ImplementationRunConfig:
    """
    Specification of an engine-specific implementation run for a given workload.
    """
    engine: str  # one of ALL_ENGINES
    timing_boundary: str  # "cpu_host_timer" or "gpu_cuda_event"
    warmup_iters: int = 5
    repeat_iters: int = 20
    block_size: int = 256
    num_streams: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImplementationRunConfig":
        return cls(**data)

    @classmethod
    def for_engine(
        cls,
        engine: str,
        warmup_iters: int = 5,
        repeat_iters: int = 20,
        block_size: int = 256,
        num_streams: int = 4,
    ) -> "ImplementationRunConfig":
        if engine not in ALL_ENGINES:
            raise ValueError(f"Unknown engine '{engine}'. Expected one of: {ALL_ENGINES}")
        boundary = TIMING_BOUNDARIES[engine]
        return cls(
            engine=engine,
            timing_boundary=boundary,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
            block_size=block_size,
            num_streams=num_streams,
        )


@dataclass
class BenchmarkSuiteProfile:
    """
    A curated suite of workload configurations and implementation targets.
    """
    profile_name: str
    description: str
    workloads: List[WorkloadConfig]
    implementations: List[str]
    warmup_iters: int = 5
    repeat_iters: int = 20

    def get_counts(self) -> Dict[str, int]:
        """Returns rigorous counts separating workloads, planned runs, and repetitions."""
        cpu_engines = [e for e in self.implementations if e in CPU_ENGINES]
        gpu_engines = [e for e in self.implementations if e in CUDA_ENGINES]

        workload_count = len(self.workloads)
        planned_cpu_runs = workload_count * len(cpu_engines)
        planned_gpu_runs = workload_count * len(gpu_engines)
        total_planned_runs = planned_cpu_runs + planned_gpu_runs

        total_cpu_repetitions = planned_cpu_runs * self.repeat_iters
        total_gpu_repetitions = planned_gpu_runs * self.repeat_iters
        total_repetitions = total_cpu_repetitions + total_gpu_repetitions

        return {
            "workload_count": workload_count,
            "cpu_engine_count": len(cpu_engines),
            "gpu_engine_count": len(gpu_engines),
            "planned_cpu_runs": planned_cpu_runs,
            "planned_gpu_runs": planned_gpu_runs,
            "total_planned_runs": total_planned_runs,
            "warmup_iters_per_run": self.warmup_iters,
            "repeat_iters_per_run": self.repeat_iters,
            "total_cpu_repetitions": total_cpu_repetitions,
            "total_gpu_repetitions": total_gpu_repetitions,
            "total_repetitions": total_repetitions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "description": self.description,
            "workloads": [w.to_dict() for w in self.workloads],
            "implementations": list(self.implementations),
            "warmup_iters": self.warmup_iters,
            "repeat_iters": self.repeat_iters,
            "counts": self.get_counts(),
        }


def get_profile_counts(profile_name: str) -> Dict[str, int]:
    """Helper returning counts for a profile by name."""
    profile = build_profile(profile_name)
    return profile.get_counts()


def build_profile(
    profile_name: str,
    warmup_iters: int = 5,
    repeat_iters: int = 20,
    implementations: Optional[List[str]] = None,
    custom_workloads: Optional[List[WorkloadConfig]] = None,
) -> BenchmarkSuiteProfile:
    """
    Builds a benchmark suite profile based on predefined matrix configurations.

    Profiles:
    - 'smoke': 6 workloads (toy x 3 ops + tier_a x 3 ops), 6 implementations (12 CPU runs, 24 GPU runs)
    - 'scaling_features': Tier A graph varying feature dimensions D in [16, 32, 64, 128, 256], ops ['mean', 'max']
    - 'scaling_graphs': Varying scale Tier A, Tier B, Tier C, ops ['mean', 'max']
    - 'power_law_vs_uniform': Power-law vs uniform topology, ops ['mean', 'sum', 'max']
    - 'custom': Custom workload list
    """
    if implementations is None:
        implementations = list(ALL_ENGINES)

    for eng in implementations:
        if eng not in ALL_ENGINES:
            raise ValueError(f"Unknown engine '{eng}' in implementations list")

    p_norm = profile_name.lower().strip()

    if p_norm == "smoke":
        # Exactly 6 logical workloads using the authoritative committed fixtures:
        # 1-3: toy_graph_5nodes (N=5, M=8, D=4) with mean, sum, max (ADR-008/009 fixture)
        # 4-6: synthetic_tier_a_10k (N=10,000, M=94,945, D=32) with mean, sum, max (Phase 5/6 baseline fixture)
        ops = ["mean", "sum", "max"]
        workloads: List[WorkloadConfig] = []
        for op in ops:
            workloads.append(
                WorkloadConfig(
                    name=f"toy_graph_5nodes_{op}",
                    graph_fixture="data/fixtures/toy_graph_5nodes.npz",
                    num_nodes=5,
                    num_edges=8,
                    feature_dim=4,
                    operator=op,
                    is_directed=True,
                    topology="toy",
                    seed=42,
                )
            )
        for op in ops:
            workloads.append(
                WorkloadConfig(
                    name=f"synthetic_tier_a_10k_{op}",
                    graph_fixture="data/fixtures/synthetic_tier_a_10k.npz",
                    num_nodes=10000,
                    num_edges=94945,
                    feature_dim=32,
                    operator=op,
                    is_directed=True,
                    topology="power_law",
                    seed=42,
                )
            )

        return BenchmarkSuiteProfile(
            profile_name="smoke",
            description="Phase 8 smoke test profile: 6 workloads x 6 engines = 36 planned runs (12 CPU, 24 GPU)",
            workloads=workloads,
            implementations=implementations,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

    elif p_norm == "scaling_features":
        # Feature dimension scaling on Tier A graph (N=10k, M=100k)
        dims = [16, 32, 64, 128, 256]
        ops = ["mean", "max"]
        workloads = []
        for d in dims:
            for op in ops:
                workloads.append(
                    WorkloadConfig(
                        name=f"tier_a_d{d}_{op}",
                        graph_fixture=None,  # Dynamically generated with dimension d
                        num_nodes=10000,
                        num_edges=100000,
                        feature_dim=d,
                        operator=op,
                        is_directed=True,
                        topology="power_law",
                        seed=42,
                        generator_params={"avg_degree": 10, "gamma": 2.5},
                    )
                )
        return BenchmarkSuiteProfile(
            profile_name="scaling_features",
            description="Feature dimension scaling suite (D in [16, 32, 64, 128, 256], ops ['mean', 'max'])",
            workloads=workloads,
            implementations=implementations,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

    elif p_norm == "scaling_graphs":
        # Graph scale scaling: Tier A, Tier B, Tier C (D=64)
        tiers = [
            ("tier_a_10k", 10000, 100000, "data/fixtures/synthetic_tier_a_10k.npz"),
            ("tier_b_100k", 100000, 1000000, "data/fixtures/synthetic_tier_b_100k.npz"),
            ("tier_c_1m", 1000000, 10000000, "data/fixtures/synthetic_tier_c_1m.npz"),
        ]
        ops = ["mean", "max"]
        workloads = []
        for name, n, m, fixture in tiers:
            for op in ops:
                workloads.append(
                    WorkloadConfig(
                        name=f"{name}_{op}",
                        graph_fixture=fixture,
                        num_nodes=n,
                        num_edges=m,
                        feature_dim=64,
                        operator=op,
                        is_directed=True,
                        topology="power_law",
                        seed=42,
                    )
                )
        return BenchmarkSuiteProfile(
            profile_name="scaling_graphs",
            description="Graph scale scaling suite (Tier A 10K, Tier B 100K, Tier C 1M, D=64, ops ['mean', 'max'])",
            workloads=workloads,
            implementations=implementations,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

    elif p_norm == "power_law_vs_uniform":
        # Topology comparison: power-law vs uniform degree distribution (N=10k, M=100k, D=64)
        topologies = ["power_law", "uniform"]
        ops = ["mean", "sum", "max"]
        workloads = []
        for top in topologies:
            for op in ops:
                fixture = "data/fixtures/synthetic_tier_a_10k.npz" if top == "power_law" else None
                workloads.append(
                    WorkloadConfig(
                        name=f"topology_{top}_{op}",
                        graph_fixture=fixture,
                        num_nodes=10000,
                        num_edges=100000,
                        feature_dim=64,
                        operator=op,
                        is_directed=True,
                        topology=top,
                        seed=42,
                        generator_params={"avg_degree": 10, "gamma": 2.5 if top == "power_law" else None},
                    )
                )
        return BenchmarkSuiteProfile(
            profile_name="power_law_vs_uniform",
            description="Topology comparison suite (power-law vs uniform, N=10K, M=100K, D=64, ops ['mean', 'sum', 'max'])",
            workloads=workloads,
            implementations=implementations,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )

    elif p_norm == "custom":
        if not custom_workloads:
            raise ValueError("Profile 'custom' requires 'custom_workloads' list")
        return BenchmarkSuiteProfile(
            profile_name="custom",
            description=f"Custom user-configured benchmark profile with {len(custom_workloads)} workloads",
            workloads=custom_workloads,
            implementations=implementations,
            warmup_iters=warmup_iters,
            repeat_iters=repeat_iters,
        )
    else:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available profiles: 'smoke', 'scaling_features', "
            "'scaling_graphs', 'power_law_vs_uniform', 'custom'"
        )
