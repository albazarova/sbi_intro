import torch
import numpy as np
import subprocess
import pickle
from PIL import Image
import argparse
import os

from sbi.utils import BoxUniform
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.utils.simulation_utils import simulate_for_sbi


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=48, type=int, help="number of workers")
    parser.add_argument(
        "--sim_type",
        default="observables",
        type=str,
        choices=["observables", "image", "both"],
        help="Simulation type. Can be one of ['observables', 'image', 'both'].",
    )
    parser.add_argument(
        "--seed", default=-1, type=int, help="random seed. -1 for no seed."
    )
    parser.add_argument(
        "--num_simulations",
        default=2000,
        type=int,
        help="number of simulations to include in database.",
    )
    args = parser.parse_args()
    return args


def simulator(pvec, sim_type="observables"):
    """
    The simulator TwoCell has 16 inputs.
    pvec can be any combination of the parameters listed.
    pvec : torch.tensor, np.array, list
    sim_type: one of ["observables", "image", "both"]
    """

    if type(pvec) == torch.Tensor:
        pvec = pvec.numpy()

    params = {"general": {}, "tumor": {}, "normal": {}}

    params["general"]["random_seed"] = 23
    params["general"]["grid_constant"] = 5
    params["general"]["radius_volume"] = 160
    params["general"]["simulation_duration"] = 100
    params["general"]["time_step"] = 0.002
    params["general"]["width_phases"] = 0.5

    params["tumor"]["initial_number"] = 1
    params["tumor"]["cycle_duration"] = pvec[0]
    params["tumor"]["mean_velocity"] = pvec[1]
    params["tumor"]["persistence_polarity"] = 1.5
    params["tumor"]["initial_position"] = 137312

    params["normal"]["initial_number"] = 5
    params["normal"]["cycle_duration"] = pvec[2]
    params["normal"]["mean_velocity"] = pvec[3]
    params["normal"]["persistence_polarity"] = 1.5
    params["normal"]["initial_position"] = 133087

    pvec = []
    for ptype in params:
        for k, v in params[ptype].items():
            pvec.append(str(v))

    module_dir = os.path.dirname(os.path.abspath(__file__))
    binary_path = os.path.join(module_dir, "TwoCell")
    result = subprocess.run([binary_path, *pvec], capture_output=True, text=True)
    if result.returncode == 0:
        array, mean_pos, radius_conf, num_cells = result.stdout.split("/**/")

        # observables
        if sim_type in ["observables", "both"]:
            mean_pos = np.fromstring(mean_pos, sep=" ")  # (6,) = (3,2)
            radius_conf = np.fromstring(radius_conf, sep=" ")  # (6,) = (3,2)
            num_cells = np.fromstring(num_cells, sep=" ")  # (2,)
        if sim_type in ["image", "both"]:
            # Last image
            array = np.fromstring(array, sep=" ").reshape(65, 65, 3)

        if sim_type == "observables":
            return torch.tensor(
                np.concatenate([mean_pos, radius_conf, num_cells], axis=0)
            )
        elif sim_type == "image":
            return torch.tensor(array)
            # return im.resize((600,600), Image.NEAREST)
        elif sim_type == "both":
            return [array, (mean_pos, radius_conf, num_cells)]
        else:
            print("Sim type unkown. exiting")
            exit()
    else:
        print("Error:\n", result.stderr)
        return torch.tensor(None)


class FlexibleBoxUniform(BoxUniform):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.device = None

    def to(self, device):
        self.device = device
        super().__init__(low=self.low, high=self.high, device=self.device)


def initialize_sbi(args):
    # example for:
    # theta -> (cycle_duration_tumor, mean_velocity_tumor, cycle_duration_normal, mean_velocity_normal)
    prior = FlexibleBoxUniform(
        low=torch.tensor([7, 0.12, 12, 0.12]), high=torch.tensor([20, 0.83, 30, 0.83])
    )
    prior.to("cpu")

    prior, theta_numel, prior_returns_numpy = process_prior(prior)
    simul = process_simulator(
        lambda theta: simulator(theta, sim_type=args.sim_type),
        prior,
        prior_returns_numpy,
    )
    return prior, simul


if __name__ == "__main__":

    args = parse_arguments()

    if args.seed != -1:
        seed_all_backends(args.seed)

    prior, simul = initialize_sbi(args)

    theta, x = simulate_for_sbi(
        simul, prior, args.num_simulations, num_workers=args.workers
    )

    with open(f"cellgrowth_sim_{args.sim_type}.pkl", "wb") as pf:
        pickle.dump((theta, x), pf)
