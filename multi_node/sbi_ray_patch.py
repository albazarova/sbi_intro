"""Ray backend for sbi's public ``simulate_for_sbi`` helper.

This is a compatibility layer, not a modification of site-packages. Local
execution delegates to sbi; Ray execution mirrors sbi's public helper while
avoiding a joblib option unsupported by Ray. The returned ``(theta, x)``
tensors follow the normal sbi contract.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch
from joblib import Parallel, delayed, parallel_backend
from sbi.utils.sbiutils import seed_all_backends

try:
    from sbi.inference import simulate_for_sbi as _sbi_simulate_for_sbi
except ImportError:  # compatibility with sbi versions that moved the helper
    from sbi.utils.simulation_utils import (
        simulate_for_sbi as _sbi_simulate_for_sbi,
    )

_RAY_JOBLIB_REGISTERED = False


def connect_ray(address: str | None = None) -> dict[str, float]:
    """Connect to the allocated Ray cluster and register its joblib backend."""
    global _RAY_JOBLIB_REGISTERED
    import ray
    from ray.util.joblib import register_ray

    if not ray.is_initialized():
        selected_address = address or os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=selected_address, ignore_reinit_error=True)
    if not _RAY_JOBLIB_REGISTERED:
        register_ray()
        _RAY_JOBLIB_REGISTERED = True
    return {key: float(value) for key, value in ray.cluster_resources().items()}


def simulate_for_sbi(
    simulator: Callable,
    proposal: Any,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: int | None = 1,
    seed: int | None = None,
    show_progress_bar: bool = True,
    ray_on: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sbi simulations locally or through Ray's distributed joblib backend.

    For local execution this delegates to the installed sbi version. For Ray,
    it mirrors sbi's sampling, batching and seeding but deliberately omits
    ``Parallel(return_as="generator")`` because Ray's joblib backend does not
    support joblib's streaming-generator mode.
    """
    if not ray_on:
        return _sbi_simulate_for_sbi(
            simulator=simulator,
            proposal=proposal,
            num_simulations=num_simulations,
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            seed=seed,
            show_progress_bar=show_progress_bar,
        )

    resources = connect_ray()
    available_cpus = max(1, int(resources.get("CPU", 1)))
    if num_workers == -1:
        num_workers = available_cpus
    num_workers = max(1, int(num_workers))
    print(
        f"Ray joblib backend: {num_workers} workers; "
        f"cluster resources={resources}",
        flush=True,
    )

    if num_simulations == 0:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty

    seed_all_backends(seed)
    theta = proposal.sample((num_simulations,))
    if simulation_batch_size is None:
        simulation_batch_size = num_simulations
    simulation_batch_size = max(
        1, min(int(simulation_batch_size), num_simulations)
    )

    # Ray's joblib backend expects numpy payloads. Exact slicing is preferable
    # to np.array_split here: every remote call receives at most the requested
    # simulation_batch_size.
    theta_numpy = theta.detach().cpu().numpy()
    batches = [
        theta_numpy[start : start + simulation_batch_size]
        for start in range(0, num_simulations, simulation_batch_size)
    ]
    batch_seeds = np.random.randint(0, 1_000_000, size=len(batches))

    def simulator_seeded(theta_batch: np.ndarray, batch_seed: int) -> torch.Tensor:
        seed_all_backends(int(batch_seed))
        return simulator(theta_batch)

    print(
        f"Submitting {len(batches)} simulation batches to Ray "
        f"(batch size <= {simulation_batch_size})",
        flush=True,
    )
    try:
        with parallel_backend("ray", n_jobs=num_workers):
            # Do not add return_as="generator": RayBackend rejects it.
            simulation_outputs = Parallel(n_jobs=num_workers)(
                delayed(simulator_seeded)(batch, batch_seed)
                for batch, batch_seed in zip(batches, batch_seeds)
            )
    except TypeError as error:
        raise TypeError(
            "The distributed simulator must accept a numpy parameter batch "
            "and return a torch.Tensor."
        ) from error

    x = torch.cat(
        [torch.as_tensor(output, dtype=torch.float32) for output in simulation_outputs],
        dim=0,
    )
    return theta.detach().cpu().to(dtype=torch.float32), x

