#!/usr/bin/env python3
"""Slurm-native multi-node Epydemix simulation followed by unmodified sbi.

The program has two stages:

1. ``simulate`` is launched with one Slurm task per node. Each task selects a
   disjoint subset of prior samples and uses a local process pool to run
   Epydemix on all CPU cores assigned to that node. It writes one tensor shard.
2. ``train`` is launched once after the distributed ``srun`` completes. It
   validates and merges all shards and trains NPE using the public sbi API.

The global simulation index determines each trajectory's random seed. Thus a
one-node and two-node run generate the same (theta, x) dataset and differ only
in how that dataset is partitioned.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from epydemix import EpiModel
from sbi.utils import BoxUniform

try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp


N = 1_000_000
E0, I0, R0 = 3_000, 1_900, 0
ALPHA = 0.85
CHANGE_DAY = 18
NUM_DAYS = 71
START_DATE = pd.Timestamp("2020-03-01")
END_DATE = START_DATE + pd.Timedelta(days=NUM_DAYS - 1)
DAYS = np.arange(NUM_DAYS)

LOWER = torch.tensor([0.8, -0.3, -0.3], dtype=torch.float32)
UPPER = torch.tensor([1.5, 0.1, 0.1], dtype=torch.float32)

INITIAL_CONDITIONS = {
    "Susceptible": np.array([N - E0 - I0 - R0]),
    "Exposed": np.array([E0]),
    "Infected": np.array([I0]),
    "Recovered": np.array([R0]),
}

# First 71 observations from sbi_intro/data/RKI_data.csv. Embedding them keeps
# this two-file cluster example independent of network access and extra files.
RKI_CONFIRMED_71 = np.array(
    [
        1902.4285714286, 2206.5714285714, 2575.8571428571,
        3034.7142857143, 3619.0, 4415.0, 5491.0, 6912.8571428571,
        8721.5714285714, 10993.8571428571, 13707.5714285714,
        16841.2857142857, 20505.1428571429, 24549.1428571429,
        28924.0, 33503.2857142857, 38265.8571428571,
        43075.4285714286, 47794.4285714286, 52439.8571428571,
        56958.7142857143, 61394.1428571429, 65782.7142857143,
        70062.2857142857, 74301.5714285714, 78484.7142857143,
        82580.5714285714, 86664.7142857143, 90732.2857142857,
        94794.8571428571, 98846.7142857143, 102799.7142857143,
        106698.2857142857, 110474.4285714286, 114176.8571428571,
        117736.4285714286, 121161.0, 124328.4285714286,
        127361.8571428571, 130289.5714285714, 132971.4285714286,
        135455.4285714286, 137765.8571428571, 139913.7142857143,
        142007.2857142857, 144005.5714285714, 145917.4285714286,
        147805.8571428571, 149615.8571428571, 151349.8571428571,
        153001.4285714286, 154548.2857142857, 156019.4285714286,
        157418.1428571429, 158752.5714285714, 160025.1428571429,
        161209.7142857143, 162332.1428571429, 163370.7142857143,
        164347.0, 165294.5714285714, 166193.4285714286,
        167065.8571428571, 167913.1428571429, 168719.1428571429,
        169516.0, 170300.7142857143, 171064.4285714286,
        171806.7142857143, 172512.5714285714, 173188.8571428571,
    ],
    dtype=np.float32,
)


def build_seir_model(beta_before: float, beta_after: float, gamma: float) -> EpiModel:
    """Construct the time-varying Epydemix SEIR model used in the notebook."""
    transmission_rate = np.where(DAYS < CHANGE_DAY, beta_before, beta_after)
    model = EpiModel(
        name="change-point SEIR",
        compartments=["Susceptible", "Exposed", "Infected", "Recovered"],
        parameters={
            "transmission_rate": transmission_rate,
            "incubation_rate": ALPHA,
            "recovery_rate": gamma,
        },
        default_population_size=N,
    )
    model.add_transition(
        source="Susceptible",
        target="Exposed",
        params=("transmission_rate", "Infected"),
        kind="mediated",
    )
    model.add_transition(
        source="Exposed",
        target="Infected",
        params="incubation_rate",
        kind="spontaneous",
    )
    model.add_transition(
        source="Infected",
        target="Recovered",
        params="recovery_rate",
        kind="spontaneous",
    )
    return model


def simulate_indexed(task: tuple[np.ndarray, int, int]) -> np.ndarray:
    """Simulate one parameter vector with a node-count-independent RNG seed."""
    theta, global_index, base_seed = task
    beta1, delta_beta, delta_gamma = np.asarray(theta, dtype=float).reshape(3)
    beta2 = max(beta1 + delta_beta, 1e-4)
    gamma = max(beta1 + delta_gamma, 1e-4)

    seed_sequence = np.random.SeedSequence([base_seed, global_index])
    rng = np.random.default_rng(seed_sequence)
    model = build_seir_model(beta1, beta2, gamma)
    results = model.run_simulations(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_conditions_dict=INITIAL_CONDITIONS,
        Nsim=1,
        resample_frequency=None,
        rng=rng,
    )
    trajectory = results.trajectories[0]
    evolved = (
        trajectory.compartments["Infected_total"]
        + trajectory.compartments["Recovered_total"]
    )
    # Epydemix stores post-step states. Restore the supplied day-zero state.
    confirmed_proxy = np.concatenate(([I0 + R0], evolved[:-1]))
    if confirmed_proxy.shape != (NUM_DAYS,):
        raise RuntimeError(f"Unexpected simulator shape: {confirmed_proxy.shape}")
    return confirmed_proxy.astype(np.float32, copy=False)


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def distributed_context() -> tuple[int, int, int, str]:
    """Return Slurm rank, number of Slurm tasks, local CPUs, and hostname."""
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    hostname = os.uname().nodename
    return rank, world_size, local_cpus, hostname


def run_simulation_stage(args: argparse.Namespace) -> None:
    rank, world_size, allocated_cpus, hostname = distributed_context()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_workers is None:
        local_workers = allocated_cpus
    else:
        local_workers = args.local_workers

    torch.manual_seed(args.seed)
    prior_cpu = BoxUniform(low=LOWER, high=UPPER)
    # This table is small. Recreating it identically on every node avoids a
    # shared-file coordination step before the distributed simulation begins.
    theta_all = prior_cpu.sample((args.num_simulations,)).cpu()
    indices = torch.arange(args.num_simulations, dtype=torch.long)[rank::world_size]
    theta_local = theta_all[indices]
    local_workers = max(1, min(local_workers, len(theta_local)))

    print(
        f"[rank {rank}/{world_size} on {hostname}] "
        f"simulating {len(indices)} trajectories with {local_workers} workers",
        flush=True,
    )
    tasks = [
        (theta.numpy(), int(index), args.seed)
        for theta, index in zip(theta_local, indices)
    ]
    chunksize = args.chunksize
    if chunksize is None:
        chunksize = max(1, len(tasks) // max(1, local_workers * 8))

    started = time.perf_counter()
    if local_workers == 1:
        simulations = [simulate_indexed(task) for task in tasks]
    else:
        with mp.Pool(processes=local_workers) as pool:
            simulations = list(pool.imap(simulate_indexed, tasks, chunksize=chunksize))
    elapsed = time.perf_counter() - started
    x_local = torch.as_tensor(np.stack(simulations), dtype=torch.float32)

    shard_path = output_dir / f"shard_{rank:04d}.pt"
    atomic_torch_save(
        {
            "indices": indices,
            "theta": theta_local,
            "x": x_local,
            "rank": rank,
            "world_size": world_size,
            "hostname": hostname,
            "elapsed_seconds": elapsed,
            "base_seed": args.seed,
            "num_simulations_total": args.num_simulations,
        },
        shard_path,
    )
    print(
        f"[rank {rank}] wrote {shard_path}; elapsed={elapsed:.3f}s; "
        f"throughput={len(indices) / elapsed:.1f} simulations/s",
        flush=True,
    )


def load_shard(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only was added
        return torch.load(path, map_location="cpu")


def merge_shards(input_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    paths = sorted(input_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {input_dir}")
    shards = [load_shard(path) for path in paths]
    indices = torch.cat([shard["indices"] for shard in shards])
    theta = torch.cat([shard["theta"] for shard in shards])
    x = torch.cat([shard["x"] for shard in shards])
    order = torch.argsort(indices)
    indices, theta, x = indices[order], theta[order], x[order]

    expected_total = int(shards[0]["num_simulations_total"])
    expected_indices = torch.arange(expected_total, dtype=torch.long)
    if not torch.equal(indices, expected_indices):
        raise RuntimeError(
            "Shard validation failed: simulation indices are missing or duplicated."
        )
    if theta.shape != (expected_total, 3) or x.shape != (expected_total, NUM_DAYS):
        raise RuntimeError(f"Unexpected merged shapes: theta={theta.shape}, x={x.shape}")

    metadata = {
        "num_simulations": expected_total,
        "num_shards": len(shards),
        "nodes": sorted({str(shard["hostname"]) for shard in shards}),
        "per_shard_elapsed_seconds": {
            str(shard["rank"]): float(shard["elapsed_seconds"]) for shard in shards
        },
        "distributed_elapsed_estimate_seconds": max(
            float(shard["elapsed_seconds"]) for shard in shards
        ),
        "seed": int(shards[0]["base_seed"]),
    }
    return theta.float(), x.float(), metadata


def posterior_pairplot(samples: torch.Tensor, destination: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sbi.analysis import pairplot

    kwargs = {
        "samples": samples,
        "limits": [[0.8, 1.5], [-0.3, 0.1], [-0.3, 0.1]],
        "labels": [r"$\beta_1$", r"$\beta_2-\beta_1$", r"$\gamma-\beta_1$"],
        "diag": "kde",
        "upper": "kde",
    }
    try:
        figure, _ = pairplot(**kwargs, figsize=(7, 7))
    except TypeError:
        figure, _ = pairplot(**kwargs, fig_kwargs={"fig_size": (7, 7)})
    figure.suptitle(title, y=1.01)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_one_posterior(
    theta: torch.Tensor,
    features: torch.Tensor,
    observation: torch.Tensor,
    device: str,
    training_batch_size: int,
    num_posterior_samples: int,
) -> tuple[object, torch.Tensor, float]:
    try:
        from sbi.inference import NPE as PosteriorInference
    except ImportError:
        from sbi.inference import SNPE as PosteriorInference

    prior_training = BoxUniform(low=LOWER.to(device), high=UPPER.to(device))
    inference = PosteriorInference(prior=prior_training, device=device)
    started = time.perf_counter()
    density_estimator = inference.append_simulations(
        theta.to(device), features.to(device)
    ).train(training_batch_size=min(training_batch_size, len(theta)))
    posterior = inference.build_posterior(density_estimator)
    samples = posterior.sample(
        (num_posterior_samples,), x=observation.to(device)
    ).detach().cpu()
    elapsed = time.perf_counter() - started
    return posterior, samples, elapsed


def run_training_stage(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    theta, x, metadata = merge_shards(input_dir)
    atomic_torch_save(
        {"theta": theta, "x": x, "metadata": metadata},
        input_dir / "training_pairs.pt",
    )

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    print(
        f"Merged theta={tuple(theta.shape)}, x={tuple(x.shape)} from "
        f"{metadata['num_shards']} node shard(s); training on {device}",
        flush=True,
    )

    y_o = torch.as_tensor(RKI_CONFIRMED_71, dtype=torch.float32)
    modes: Iterable[str]
    modes = ("trajectory", "mse") if args.train_mode == "both" else (args.train_mode,)
    posterior_mean_curves: dict[str, np.ndarray] = {}
    training_times: dict[str, float] = {}

    for mode in modes:
        if mode == "trajectory":
            features = x
            observation = y_o
        else:
            features = torch.mean((x - y_o.reshape(1, -1)) ** 2, dim=1, keepdim=True)
            observation = torch.zeros(1, dtype=torch.float32)

        posterior, samples, elapsed = train_one_posterior(
            theta=theta,
            features=features,
            observation=observation,
            device=device,
            training_batch_size=args.training_batch_size,
            num_posterior_samples=args.posterior_samples,
        )
        training_times[mode] = elapsed
        torch.save(samples, input_dir / f"posterior_samples_{mode}.pt")
        posterior_pairplot(
            samples,
            input_dir / f"pairplot_{mode}.png",
            f"Epydemix + sbi ({mode})",
        )
        try:
            with (input_dir / f"posterior_{mode}.pkl").open("wb") as handle:
                pickle.dump(posterior, handle)
        except Exception as error:
            print(f"Warning: could not pickle {mode} posterior: {error}", flush=True)

        mean_theta = samples.mean(dim=0).numpy()
        posterior_mean_curves[mode] = simulate_indexed(
            (mean_theta, args.posterior_seed_offset, metadata["seed"])
        )
        print(f"Finished {mode} inference in {elapsed:.1f}s", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for mode, curve in posterior_mean_curves.items():
        axis.plot(curve, label=f"posterior mean: {mode}")
    axis.plot(RKI_CONFIRMED_71, color="black", linewidth=2, label="RKI data")
    axis.set(xlabel="Day", ylabel="Confirmed-case proxy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(input_dir / "simulated_vs_RKI.png", dpi=160)
    plt.close(figure)

    metadata.update(
        {
            "device": device,
            "train_mode": args.train_mode,
            "training_elapsed_seconds": training_times,
            "torch_version": torch.__version__,
        }
    )
    try:
        import epydemix
        import sbi

        metadata["epydemix_version"] = epydemix.__version__
        metadata["sbi_version"] = getattr(sbi, "__version__", "unknown")
    except Exception:
        pass
    (input_dir / "run_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Results written to {input_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="write this node's shard")
    simulate_parser.add_argument("--num-simulations", type=int, default=50_000)
    simulate_parser.add_argument("--output-dir", required=True)
    simulate_parser.add_argument("--seed", type=int, default=2026)
    simulate_parser.add_argument("--local-workers", type=int)
    simulate_parser.add_argument("--chunksize", type=int)

    train_parser = subparsers.add_parser("train", help="merge shards and run sbi")
    train_parser.add_argument("--input-dir", required=True)
    train_parser.add_argument(
        "--train-mode", choices=("trajectory", "mse", "both"), default="both"
    )
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--training-batch-size", type=int, default=1_000)
    train_parser.add_argument("--posterior-samples", type=int, default=10_000)
    train_parser.add_argument("--posterior-seed-offset", type=int, default=10_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "simulate":
        run_simulation_stage(args)
    else:
        run_training_stage(args)


if __name__ == "__main__":
    main()
