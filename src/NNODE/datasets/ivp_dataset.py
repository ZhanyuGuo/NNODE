import os
import sys

file_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(root_dir)


import matplotlib.pyplot as plt
import numpy as np
import ode_func
import torch
from loguru import logger
from options import Options
from scipy.integrate import odeint
from tqdm import tqdm
from utils import plot_inverted_pendulum


class IvpDataset(torch.utils.data.Dataset):
    def __init__(self, opts, is_train=True):
        super().__init__()

        logger.info(f"dataset: {self.__class__.__name__}, ode_func: {opts.ode_func}")

        self.opts = opts
        self.data = []

        if not is_train:
            self.opts.num_curves = 100
            self.opts.num_samples_per_curve = 66

        logger.info("making data ...")
        for _ in tqdm(range(self.opts.num_curves)):
            self.makeData()

        logger.info("finished")

    def makeData(self):
        # time arange
        t = np.linspace(self.opts.t_0, self.opts.t_f, self.opts.num_samples_per_curve)

        # initial state: standard normal
        # x0 = np.random.randn(2 * self.opts.num_vars * self.opts.order_s) * self.opts.scale
        x0 = np.random.uniform(-1.0, 1.0, 2 * self.opts.num_vars * self.opts.order_s) * self.opts.scale

        # for odePolyVel
        if self.opts.ode_func == "odePolyVel":
            x0[0] = x0[-1] = 0.0  # inital pos and jerk are 0

        # const input
        const_input = x0

        # solve ode
        if self.opts.ode_func == "odeVdpMu":
            mu = np.random.uniform(-1.0, 1.0) * self.opts.mu_max
            const_input = np.append(const_input, mu)
            (xt, output) = odeint(getattr(ode_func, self.opts.ode_func), x0, t, args=(mu,), full_output=1)
        else:
            (xt, output) = odeint(getattr(ode_func, self.opts.ode_func), x0, t, full_output=1)

        # integration failure, retry
        if output["message"] != "Integration successful.":
            self.makeData()
            return

        # # plot
        # plt.plot(t, xt[:, 0])
        # plt.show()
        # plot_inverted_pendulum(t, xt)

        for i in range(self.opts.num_samples_per_curve):
            input = torch.from_numpy(np.append(const_input, t[i])).float()
            label = torch.from_numpy(xt[i, :]).float()
            self.data.append((input, label))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    options = Options()
    opts = options.parse()

    ivp_dataset = IvpDataset(opts, is_train=False)
    print(len(ivp_dataset))
