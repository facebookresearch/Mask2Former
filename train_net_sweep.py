import warnings

try:
    import job_configs
except:
    warnings.warn("job_configs.py is needed for launching experiments on toolkit")

import argparse
import train_net
import pickle

from exp_configs import EXP_GROUPS
from pathlib import Path
from haven import haven_wizard as hw
from haven import haven_utils as hu


def main_sweep(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    savedir = Path(savedir)

    # Convert exp_dict dictionary to arguments
    cfg = hu.load_pkl(f"{exp_dict['config_path']}")
    # Adjust Path
    cfg.defrost()
    cfg.OUTPUT_DIR = f"{savedir}"
    cfg.DATALOADER.NUM_WORKERS = 0
    # Call main with these arguments
    train_net.main_launch(cfg, num_gpus=exp_dict["num_gpus"])

    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Choose Job Scheduler."
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, others = parser.parse_known_args()

    # Define a list of experiments as the arguments that would be passed to main
    exp_list = EXP_GROUPS[args.exp_group]

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "toolkit":
        job_config = job_configs.JOB_CONFIG

    # Run experiments and create results file
    hw.run_wizard(
        func=main_sweep,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results.ipynb",
        python_binary_path=args.python_binary,
        args=args,
    )
