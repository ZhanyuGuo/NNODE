import argparse
import os


class Options:
    def __init__(self):
        # get the root directory
        file_dir = os.path.dirname(__file__)
        root_dir = os.path.abspath(os.path.join(file_dir, os.pardir, os.pardir))

        # create the parser
        self.parser = argparse.ArgumentParser()

        # path
        self.parser.add_argument("--log_dir", type=str, default=os.path.join(root_dir, "checkpoints"))
        self.parser.add_argument("--problem_type", type=str, default="IVP", choices=["IVP", "BVP"])
        self.parser.add_argument("--checkpoint_name", type=str, default="2025-04-13-13-22-05")

        # problem
        self.parser.add_argument("--ode_func", type=str, default="odeVdpMu", help="function name in ode_func.py")
        self.parser.add_argument("--order_s", type=int, default=1, help="s in problem formulation, i.e., 2s is the order of ode")
        self.parser.add_argument("--num_vars", type=int, default=1, help="number of variables in ode")
        self.parser.add_argument("--t_0", type=float, default=-2.0, help="initial time")
        self.parser.add_argument("--t_f", type=float, default=2.0, help="terminal time")
        self.parser.add_argument("--scale", type=float, default=1.0, help="scale of the bound")
        self.parser.add_argument("--mu_max", type=float, default=1.0, help="range of mu in vdp")

        # network
        self.parser.add_argument("--model_name", type=str, default="MLP", choices=["MLP", "MLPR"])
        self.parser.add_argument("--num_neurals", type=int, default=48)
        self.parser.add_argument("--num_layers", type=int, default=5)
        self.parser.add_argument("--one_output", type=bool, default=False, help="only one output, nested automatic differentiation")

        # dataset
        self.parser.add_argument("--num_curves", type=int, default=1000, help="number of curves in training set")
        self.parser.add_argument("--num_samples_per_curve", type=int, default=101, help="number of samples per curve in training set")

        # system
        self.parser.add_argument("--no_cuda", type=bool, default=True, help="True for cpu, False for gpu")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
        self.parser.add_argument("--random_seed", type=int, default=3407)

        # optimization
        self.parser.add_argument("--has_regression_loss", type=bool, default=False)
        self.parser.add_argument("--only_regression_loss", type=bool, default=False)
        self.parser.add_argument("--lambda_f", type=float, default=1.0, help="weight of ode loss")
        self.parser.add_argument("--lambda_b", type=float, default=1.0, help="weight of boundary loss")
        self.parser.add_argument("--lambda_r", type=float, default=1.0, help="weight of regression loss")

        # training
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--step_size", type=int, default=30)
        self.parser.add_argument("--learning_rate", type=float, default=1e-3)
        self.parser.add_argument("--save_frequency", type=int, default=20)

    def parse(self):
        return self.parser.parse_args()
