import json
import os
import time

import datasets
import networks
import torch
from loguru import logger
from tensorboardX import SummaryWriter
from utils import getLoss


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        # checkpoint name
        current_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_dir = os.path.join(self.opts.log_dir, self.opts.problem_type, current_timestamp)
        logger.info(f"save_dir: {self.save_dir}")

        # device: cuda or cpu
        self.device = torch.device("cpu" if self.opts.no_cuda or not torch.cuda.is_available() else "cuda")
        logger.info(f"device: {self.device}")

        # dataset
        datasets_dict = {"IVP": "IvpDataset", "BVP": "BvpDataset"}
        train_dataset = getattr(datasets, datasets_dict[self.opts.problem_type])(self.opts)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
            pin_memory=True,
        )

        # log configs
        num_data = len(train_dataset)
        self.num_steps_per_epoch = num_data // self.opts.batch_size + 1
        self.log_interval = self.num_steps_per_epoch // 5
        self.writer = SummaryWriter(self.save_dir)

        # loss type
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # number of inputs and outputs
        num_inputs = 2 * self.opts.num_vars * self.opts.order_s + 1  # boundary(2ns) and time(1)
        num_outputs = 2 * self.opts.num_vars * self.opts.order_s  # [0, 2ns - 1] derivatives
        if self.opts.ode_func == "odeVdpMu":
            num_inputs += 1  # mu

        if self.opts.one_output:
            num_outputs = self.opts.num_vars

        # model
        logger.info(f"model: {self.opts.model_name}")
        self.model = getattr(networks, self.opts.model_name)(
            num_inputs=num_inputs,
            num_neurals=self.opts.num_neurals,
            num_layers=self.opts.num_layers,
            num_outputs=num_outputs,
        )
        self.model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opts.learning_rate)

        # scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.opts.step_size, 0.1, verbose=True)

        # save options
        self.saveOpts()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.running_loss = {}  # record loss for each log interval
        for self.epoch in range(self.opts.num_epochs):
            self.runEpoch()
            if (self.epoch + 1) % self.opts.save_frequency == 0:
                self.saveModel()

    def runEpoch(self):
        logger.info(f"training epoch {self.epoch + 1} ...")

        for idx, (inputs, labels) in enumerate(self.train_dataloader):
            # inputs and labels
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.opts.only_regression_loss:
                losses = self.processBatch(inputs, labels)
            else:
                # get input t, set require gradient
                input_t = inputs[:, -1:]
                input_t.requires_grad_(True)

                # here we have to cat them to get the calculation graph from input_t
                inputs = torch.cat([inputs[:, :-1], input_t], dim=1)

                # process batch
                losses = self.processBatch(inputs, labels, input_t)

            # get loss
            loss = losses["loss"]

            # clear gradient
            self.optimizer.zero_grad()

            # backward gradient
            loss.backward()

            # update parameters
            self.optimizer.step()

            # record lr and loss
            self.writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0], self.step)
            for k, v in losses.items():
                self.writer.add_scalar(k, v, self.step)
                self.running_loss[k] = self.running_loss.get(k, 0) + v.item()

            # print screen
            if (idx + 1) % self.log_interval == 0:
                self.log(idx)

            self.step += 1

        # update scheduler
        self.lr_scheduler.step()

    def processBatch(self, inputs, labels, input_t=None):
        outputs = self.model(inputs)
        losses = getLoss(self.criterion, outputs, inputs, labels, self.opts, input_t)
        return losses

    def log(self, idx):
        loss_strs = ""
        for k, v in self.running_loss.items():
            loss_str = f"{k}: {v / self.log_interval:.3e}, "
            loss_strs += loss_str

        elapsed_time = time.time() - self.start_time
        num_finished_iters = self.num_steps_per_epoch * self.epoch + idx + 1
        num_left_iters = self.num_steps_per_epoch * (self.opts.num_epochs - self.epoch - 1) + (self.num_steps_per_epoch - idx - 1)
        eta = num_left_iters / num_finished_iters * elapsed_time
        time_strs = f'elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}, eta: {time.strftime("%H:%M:%S", time.gmtime(eta))}'
        logger.info(f"[{self.epoch + 1:4d}/{self.opts.num_epochs:4d}, {idx + 1:4d}/{self.num_steps_per_epoch:4d}] {loss_strs}{time_strs}")

        # reset running loss
        self.running_loss = {}

    def saveOpts(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "opt.json")

        to_save = self.opts.__dict__.copy()
        with open(save_path, "w") as f:
            json.dump(to_save, f, indent=2)

        logger.info(f"options saved as: {save_path}")

    def saveModel(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"weights_{self.epoch + 1}.pth")

        torch.save(self.model.state_dict(), save_path)

        logger.info(f"model saved as: {save_path}")
