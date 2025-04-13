import json
import os
import time

import networks
import torch
from loguru import logger
from networks.mlp import MLP


class Converter:
    def __init__(self, opts):
        self.opts = opts

        # checkpoint name
        self.save_dir = os.path.join(self.opts.log_dir, self.opts.problem_type, self.opts.checkpoint_name)
        file_path = os.path.join(self.save_dir, "weights_100.pth")
        opts_path = os.path.join(self.save_dir, "opt.json")

        with open(opts_path, "r") as f:
            self.opts.__dict__.update(json.load(f))

        # number of inputs and outputs
        num_inputs = 2 * self.opts.num_vars * self.opts.order_s + 1  # boundary(2s) and time(1)
        num_outputs = 2 * self.opts.num_vars * self.opts.order_s  # [0, 2s - 1] derivatives
        if self.opts.ode_func == "odeVdpMu":
            num_inputs += 1  # mu

        logger.info(f"model: {self.opts.model_name}")
        self.model = getattr(networks, self.opts.model_name)(
            num_inputs=num_inputs,
            num_neurals=self.opts.num_neurals,
            num_outputs=num_outputs,
            num_layers=self.opts.num_layers,
        )
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

    def run(self):
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(os.path.join(self.save_dir, "model.pt"))
        logger.info(f"output: {os.path.abspath(self.save_dir)} successfully!")
