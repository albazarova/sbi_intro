#!/usr/bin/env python3
"""Distribute Epydemix simulations with Ray, then train unmodified sbi.

The same program runs on one or several Ray nodes.  Ray handles simulation
scheduling; after it returns the normal ``(theta, x)`` tensors, sbi trains NPE
on the Ray head node.  Every Ray worker writes a small marker so the result
summary proves which physical nodes executed simulation work.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from epydemix import EpiModel
from sbi.utils import BoxUniform

from sbi_ray_patch import connect_ray, simulate_for_sbi


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

# First 71 observations used by the corresponding sbi_intro notebook.
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
        source="Susceptible", target="Exposed",
        params=("transmission_rate", "Infected"), kind="mediated",
    )
    model.add_transition(
        source="Exposed", target="Infected",
        params="incubation_rate", kind="spontaneous",
    )
    model.add_transition(
        source="Infected", target="Recovered",
        params="recovery_rate", kind="spontaneous",
    )
    return model


def simulate_one(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    beta1, delta_beta, delta_gamma = np.asarray(theta, dtype=float).reshape(3)
    beta2 = max(beta1 + delta_beta, 1e-4)
    gamma = max(beta1 + delta_gamma, 1e-4)
    results = build_seir_model(beta1, beta2, gamma).run_simulations(
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
    confirmed_proxy = np.concatenate(([I0 + R0], evolved[:-1]))
    if confirmed_proxy.shape != (NUM_DAYS,):
        raise RuntimeError(f"Unexpected Epydemix output: {confirmed_proxy.shape}")
    return confirmed_proxy.astype(np.float32, copy=False)


class EpydemixBatchSimulator:
    """Picklable batched simulator suitable for sbi + Ray Joblib workers."""

    def __init__(self, activity_dir: Path):
        self.activity_dir = str(activity_dir)

    def __call__(self, parameters) -> torch.Tensor:
        if torch.is_tensor(parameters):
            values = parameters.detach().cpu().numpy()
        else:
            values = np.asarray(parameters)
        values = np.atleast_2d(values).astype(float, copy=False)

        hostname = socket.gethostname()
        marker = Path(self.activity_dir) / f"{hostname}__pid_{os.getpid()}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"hostname": hostname, "pid": os.getpid()}))

        outputs = []
        for theta in values:
            # sbi seeds numpy separately for each distributed batch.
            trajectory_seed = int(np.random.randint(0, 2**32 - 1))
            outputs.append(simulate_one(theta, np.random.default_rng(trajectory_seed)))
        return torch.as_tensor(np.stack(outputs), dtype=torch.float32)


def wait_for_ray_nodes(expected: int, timeout: float) -> tuple[list[str], dict[str, float]]:
    import ray

    deadline = time.monotonic() + timeout
    connect_ray()
    while True:
        alive = [node for node in ray.nodes() if node.get("Alive", False)]
        hosts = sorted(
            {
                str(node.get("NodeManagerHostname") or node.get("NodeManagerAddress"))
                for node in alive
            }
        )
        if len(hosts) >= expected:
            resources = {
                key: float(value)
                for key, value in ray.cluster_resources().items()
            }
            return hosts, resources
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Ray cluster has {len(hosts)} live node(s), expected {expected}: {hosts}"
            )
        print(f"Waiting for Ray nodes: {len(hosts)}/{expected}", flush=True)
        time.sleep(2)


def summarize_activity(activity_dir: Path) -> dict[str, int]:
    workers: Counter[str] = Counter()
    for marker in activity_dir.glob("*.json"):
        payload = json.loads(marker.read_text())
        workers[str(payload["hostname"])] += 1
    return dict(sorted(workers.items()))


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


def train_posterior(
    theta: torch.Tensor,
    features: torch.Tensor,
    observation: torch.Tensor,
    device: str,
    training_batch_size: int,
    posterior_samples: int,
) -> tuple[object, torch.Tensor, float]:
    try:
        from sbi.inference import NPE as PosteriorInference
    except ImportError:
        from sbi.inference import SNPE as PosteriorInference

    training_prior = BoxUniform(low=LOWER.to(device), high=UPPER.to(device))
    inference = PosteriorInference(prior=training_prior, device=device)
    started = time.perf_counter()
    estimator = inference.append_simulations(
        theta.to(device), features.to(device)
    ).train(training_batch_size=min(training_batch_size, len(theta)))
    posterior = inference.build_posterior(estimator)
    samples = posterior.sample(
        (posterior_samples,), x=observation.to(device)
    ).detach().cpu()
    return posterior, samples, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-simulations", type=int, default=50_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--simulation-batch-size", type=int)
    parser.add_argument("--expected-ray-nodes", type=int, default=1)
    parser.add_argument("--ray-wait-seconds", type=float, default=120)
    parser.add_argument(
        "--train-mode", choices=("trajectory", "mse", "both"), default="both"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-batch-size", type=int, default=1_000)
    parser.add_argument("--posterior-samples", type=int, default=10_000)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_simulations < 1:
        raise ValueError("--num-simulations must be positive")
    if args.simulation_batch_size is not None and args.simulation_batch_size < 1:
        raise ValueError("--simulation-batch-size must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    activity_dir = output_dir / f"ray_activity_{time.time_ns()}"
    activity_dir.mkdir()

    ray_hosts, ray_resources = wait_for_ray_nodes(
        args.expected_ray_nodes, args.ray_wait_seconds
    )
    ray_cpus = max(1, int(ray_resources.get("CPU", 1)))
    batch_size = args.simulation_batch_size
    if batch_size is None:
        # About eight batches per Ray CPU gives the scheduler room to load-balance.
        batch_size = max(1, args.num_simulations // (8 * ray_cpus))
    num_batches = max(1, args.num_simulations // batch_size)
    print(
        f"Ray nodes={ray_hosts}; CPUs={ray_cpus}; simulations={args.num_simulations}; "
        f"batch_size={batch_size}; approximately {num_batches} Ray tasks",
        flush=True,
    )

    torch.manual_seed(args.seed)
    simulation_prior = BoxUniform(LOWER, UPPER)
    simulator = EpydemixBatchSimulator(activity_dir)
    simulation_started = time.perf_counter()
    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=simulation_prior,
        num_simulations=args.num_simulations,
        num_workers=ray_cpus,
        simulation_batch_size=batch_size,
        seed=args.seed,
        ray_on=True,
    )
    simulation_elapsed = time.perf_counter() - simulation_started
    theta, x = theta.cpu().float(), x.cpu().float()
    if theta.shape != (args.num_simulations, 3):
        raise RuntimeError(f"Unexpected theta shape: {tuple(theta.shape)}")
    if x.shape != (args.num_simulations, NUM_DAYS):
        raise RuntimeError(f"Unexpected x shape: {tuple(x.shape)}")

    worker_activity = summarize_activity(activity_dir)
    print("Ray workers that executed Epydemix batches:", flush=True)
    for hostname, workers in worker_activity.items():
        print(f"  {hostname}: {workers} worker process(es)", flush=True)
    if len(worker_activity) < args.expected_ray_nodes:
        raise RuntimeError(
            f"Only {len(worker_activity)} of {args.expected_ray_nodes} Ray nodes "
            "executed simulations. Reduce --simulation-batch-size or increase "
            "--num-simulations."
        )

    torch.save({"theta": theta, "x": x}, output_dir / "training_pairs.pt")
    summary: dict[str, object] = {
        "num_simulations": args.num_simulations,
        "simulation_batch_size": batch_size,
        "ray_cpus": ray_cpus,
        "ray_cluster_nodes": ray_hosts,
        "ray_resources": ray_resources,
        "simulation_worker_processes_by_node": worker_activity,
        "simulation_elapsed_seconds": simulation_elapsed,
        "seed": args.seed,
    }

    if not args.skip_training:
        if args.device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

        observation_full = torch.as_tensor(RKI_CONFIRMED_71)
        modes: Iterable[str] = (
            ("trajectory", "mse") if args.train_mode == "both" else (args.train_mode,)
        )
        training_times: dict[str, float] = {}
        mean_curves: dict[str, np.ndarray] = {}
        for mode in modes:
            if mode == "trajectory":
                features, observation = x, observation_full
            else:
                features = torch.mean(
                    (x - observation_full.reshape(1, -1)) ** 2,
                    dim=1,
                    keepdim=True,
                )
                observation = torch.zeros(1)
            posterior, samples, elapsed = train_posterior(
                theta, features, observation, device,
                args.training_batch_size, args.posterior_samples,
            )
            training_times[mode] = elapsed
            torch.save(samples, output_dir / f"posterior_samples_{mode}.pt")
            posterior_pairplot(
                samples, output_dir / f"pairplot_{mode}.png",
                f"Ray + Epydemix + sbi ({mode})",
            )
            try:
                with (output_dir / f"posterior_{mode}.pkl").open("wb") as handle:
                    pickle.dump(posterior, handle)
            except Exception as error:
                print(f"Warning: could not pickle {mode} posterior: {error}")
            mean_theta = samples.mean(dim=0).numpy()
            mean_curves[mode] = simulate_one(mean_theta, np.random.default_rng(args.seed))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 4.5))
        for mode, curve in mean_curves.items():
            axis.plot(curve, label=f"posterior mean: {mode}")
        axis.plot(RKI_CONFIRMED_71, color="black", linewidth=2, label="RKI data")
        axis.set(xlabel="Day", ylabel="Confirmed-case proxy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "simulated_vs_RKI.png", dpi=160)
        plt.close(figure)
        summary.update(
            device=device,
            train_mode=args.train_mode,
            training_elapsed_seconds=training_times,
        )

    import epydemix
    import ray
    import sbi

    summary.update(
        epydemix_version=getattr(epydemix, "__version__", "unknown"),
        ray_version=getattr(ray, "__version__", "unknown"),
        sbi_version=getattr(sbi, "__version__", "unknown"),
        torch_version=torch.__version__,
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Finished; results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
