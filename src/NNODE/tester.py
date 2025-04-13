import json
import os
from timeit import default_timer as timer

import datasets
import networks
import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp
from tqdm import tqdm
from utils import getGrad, getLoss, plot_ghost, plot_inverted_pendulum, plot_inverted_pendulum_animation, plot_inverted_pendulum_animation_compare

# plt.rcParams["font.family"] = "Times New Roman, SimSun"
plt.rcParams["font.size"] = 14
plt.rcParams["mathtext.fontset"] = "stix"

RED = "#F27970"
GREEN = "#54B345"
BLUE = "#05B9E2"


class Tester:
    def __init__(self, opts):
        self.opts = opts

        # checkpoint name
        file_name = self.opts.checkpoint_name + "/weights_100.pth"
        opts_name = self.opts.checkpoint_name + "/opt.json"
        file_path = os.path.join(self.opts.log_dir, self.opts.problem_type, file_name)
        opts_path = os.path.join(self.opts.log_dir, self.opts.problem_type, opts_name)

        with open(opts_path, "r") as f:
            self.opts.__dict__.update(json.load(f))
            self.opts.no_cuda = True
            self.opts.random_seed += 1
            self.opts.has_regression_loss = True

        # dataset
        datasets_dict = {"IVP": "IvpDataset", "BVP": "BvpDataset"}
        test_dataset = getattr(datasets, datasets_dict[self.opts.problem_type])(self.opts, is_train=False)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.opts.num_samples_per_curve,  # one curve one batch
            shuffle=False,
            num_workers=self.opts.num_workers,
        )

        # loss type
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # number of inputs and outputs
        num_inputs = 2 * self.opts.num_vars * self.opts.order_s + 1  # boundary(2ns) and time(1)
        num_outputs = 2 * self.opts.num_vars * self.opts.order_s  # [0, 2ns - 1] derivatives
        if self.opts.ode_func == "odeVdpMu":
            num_inputs += 1  # mu

        if self.opts.one_output:
            num_outputs = 1

        # network model
        logger.info(f"model: {self.opts.model_name}")
        self.model = getattr(networks, self.opts.model_name)(
            num_inputs=num_inputs,
            num_neurals=self.opts.num_neurals,
            num_layers=self.opts.num_layers,
            num_outputs=num_outputs,
        )
        self.model.load_state_dict(torch.load(file_path))
        for param in self.model.parameters():
            param.requires_grad_(False)

    def test_qualitative(self):
        l_f, l_b, l_r, loss = [], [], [], []
        for inputs, labels in tqdm(self.test_dataloader):
            # get input t, set require gradient
            input_t = inputs[:, -1:]
            input_t.requires_grad_(True)
            inputs = torch.cat([inputs[:, :-1], input_t], dim=1)

            # forward
            outputs = self.model(inputs)
            losses = getLoss(self.criterion, outputs, inputs, labels, self.opts, input_t)

            l_f.append(losses["l_f"].item())
            l_b.append(losses["l_b"].item())
            l_r.append(losses["l_r"].item())
            loss.append(losses["loss"].item())

            inputs = inputs.detach()
            outputs = outputs.detach()

            legends = ["Reference", "Proposed"]
            # legends = ["参考真值", "本章算法"]

            # plt.title(f"mu = {inputs[0, -2]:.3f}")
            plt.plot(inputs[:, -1], labels[:, 0], linestyle="-", linewidth=2, color=RED, label=legends[0])
            plt.plot(inputs[:, -1], outputs[:, 0], linestyle="--", linewidth=2, color=GREEN, label=legends[1])
            plt.xlabel(r"$t$", fontsize=20)
            plt.ylabel(r"$x(t)$", fontsize=20)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

        logger.info("l_f \t l_b \t l_r \t loss ")
        logger.info(f"{sum(l_f) / len(l_f):.3e}, {sum(l_b) / len(l_b):.3e}, {sum(l_r) / len(l_r):.3e}, {sum(loss) / len(loss):.3e}")

    def test_quantitative(self):
        assert self.opts.problem_type in ["IVP", "BVP"], "Not implemented."
        assert self.opts.order_s in [1, 2], "Not implemented."

        problem_type = self.opts.problem_type
        num_vars = self.opts.num_vars
        order_s = self.opts.order_s

        l_f, l_b, l_r, l_c, loss = [], [], [], [], []
        for inputs, labels in tqdm(self.test_dataloader):
            # all gradients need
            inputs.requires_grad_(True)
            input_t = inputs[:, -1:]
            inputs = torch.cat([inputs[:, :-1], input_t], dim=1)

            outputs = self.model(inputs)
            losses = getLoss(self.criterion, outputs, inputs, labels, self.opts, input_t)

            l_f.append(losses["l_f"].item())
            l_b.append(losses["l_b"].item())
            l_r.append(losses["l_r"].item())
            loss.append(losses["loss"].item())

            num_outputs = outputs.shape[1]
            doutputs = torch.stack([getGrad(outputs[:, i], inputs) for i in range(num_outputs)], dim=1)

            inputs = inputs.detach()
            outputs = outputs.detach()
            doutputs = doutputs.detach()

            t_0 = self.opts.t_0
            t_f = self.opts.t_f
            duration = t_f - t_0

            t_c = t_0 + 0.5 * duration
            idx_c = int((t_c - t_0) / duration * (self.opts.num_samples_per_curve - 1))
            delta_c = 0.5

            # only test the first varible
            inputs_new = inputs.clone()
            inputs_new[:, 0] += delta_c / doutputs[idx_c, 0, 0]
            outputs_new = self.model(inputs_new)
            outputs_new = outputs_new.detach()
            l_c.append((outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000)

            inputs_new = inputs.clone()
            if problem_type == "IVP":
                inputs_new[:, num_vars] += delta_c / doutputs[idx_c, 0, num_vars]
            elif problem_type == "BVP":
                inputs_new[:, num_vars * order_s] += delta_c / doutputs[idx_c, 0, num_vars * order_s]

            outputs_new = self.model(inputs_new)
            outputs_new = outputs_new.detach()
            l_c.append((outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000)

        logger.info("l_f \t l_b \t l_r \t loss \t [l_c (1e-4)] ")
        logger.info(f"{sum(l_f) / len(l_f):.3e}, {sum(l_b) / len(l_b):.3e}, {sum(l_r) / len(l_r):.3e}, {sum(loss) / len(loss):.3e}, {sum(l_c) / len(l_c):.3e}")

    def test_bound(self):
        assert self.opts.problem_type in ["IVP", "BVP"], "Not implemented."
        assert self.opts.order_s in [1, 2], "Not implemented."

        order_s = self.opts.order_s
        dataiter = iter(self.test_dataloader)
        inputs_1, _ = next(dataiter)
        inputs_2, _ = next(dataiter)

        inputs_1[:, order_s:] = inputs_2[:, order_s:]
        outputs = self.model(inputs_1)

        inputs_1 = inputs_1.detach()
        outputs = outputs.detach()

        plt.plot(inputs_1[:, -1], outputs[:, 0], color=RED)
        plt.scatter(inputs_1[0, -1], inputs_1[0, 0], color=GREEN)

        if self.opts.problem_type == "IVP":
            delta = 0.1
            plt.arrow(inputs_1[0, -1], inputs_1[0, 0], delta, delta * inputs_1[0, 1])
        elif self.opts.problem_type == "BVP":
            plt.scatter(inputs_1[-1, -1], inputs_1[-1, 1], color=GREEN)

        plt.show()

    def test_grad(self):
        assert self.opts.problem_type in ["IVP", "BVP"], "Not implemented."
        assert self.opts.order_s in [1, 2], "Not implemented."
        assert self.opts.num_vars == 1, "Not implemented."

        idx = 0
        loss = []
        head_width, head_length = 0.02, 0.05

        for inputs, _ in tqdm(self.test_dataloader):
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            # TODO: not updated
            if self.opts.one_output:
                doutputs = getGrad(outputs, inputs)

                inputs = inputs.detach()
                outputs = outputs.detach()
                doutputs = doutputs.detach()

                t_0 = self.opts.t_0
                t_f = self.opts.t_f
                duration = t_f - t_0
                t_c = t_0 + 0.6 * duration
                idx_c = int((self.opts.num_samples_per_curve - 1) * (t_c - t_0) / duration)
                delta_c = 0.5

                inputs_0 = inputs.clone()
                inputs_1 = inputs.clone()
                inputs_0[:, 0] += delta_c / doutputs[idx_c, 0]
                inputs_1[:, 1] += delta_c / doutputs[idx_c, 1]
                outputs_0 = self.model(inputs_0)
                outputs_1 = self.model(inputs_1)
                outputs_0 = outputs_0.detach()
                outputs_1 = outputs_1.detach()

                loss.append((outputs_0[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000)
                loss.append((outputs_1[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000)
            else:
                num_outputs = outputs.shape[1]
                doutputs = torch.stack([getGrad(outputs[:, i], inputs) for i in range(num_outputs)], dim=1)

                inputs = inputs.detach()
                outputs = outputs.detach()
                doutputs = doutputs.detach()

                t_0 = self.opts.t_0
                t_f = self.opts.t_f
                duration = t_f - t_0

                # t_c = t_0 + 0.5 * duration
                t_c = 0.4
                idx_c = int((t_c - t_0) / duration * (self.opts.num_samples_per_curve - 1))
                delta_c = 0.1

                legends = ["Original", "Change $x(t_0)$", "Change $x(t_f)$", "Change $x'(t_0)$", "Change $x''(t_0)$"]
                # legends = ["原有轨迹", "调整 x(-2)", "调整 x(2)", '调整 x"(0)']

                # viz specific curve
                if idx in [1, 2, 3]:
                    plt.plot(inputs[:, -1], outputs[:, 0], color=RED, label=legends[0])
                    plt.scatter(inputs[idx_c, -1], outputs[idx_c, 0], c=RED)  # point at t_c
                    plt.arrow(
                        inputs[idx_c, -1],
                        outputs[idx_c, 0],
                        0,
                        delta_c,
                        ec=RED,
                        fc=RED,
                        head_width=head_width,
                        head_length=head_length,
                        length_includes_head=True,
                    )  # arrow for delta_c

                    if self.opts.problem_type == "IVP":
                        delta = 0.1
                        plt.arrow(
                            inputs[0, -1],
                            outputs[0, 0],
                            0,
                            delta_c / doutputs[idx_c, 0, 0],
                            ec=GREEN,
                            fc=GREEN,
                            head_width=head_width,
                            head_length=head_length,
                            length_includes_head=True,
                        )  # change for x(t0)

                        plt.arrow(
                            inputs[0, -1],
                            outputs[0, 0],
                            delta,
                            delta * inputs[0, 1],
                            ec=RED,
                            fc=RED,
                            head_width=head_width,
                            head_length=head_length,
                            length_includes_head=True,
                        )  # x'(t0)

                        plt.arrow(
                            inputs[0, -1] + delta,
                            outputs[0, 0] + delta * inputs[0, 1],
                            0,
                            delta * delta_c / doutputs[idx_c, 0, 1],
                            ec=BLUE,
                            fc=BLUE,
                            head_width=head_width,
                            head_length=head_length,
                            length_includes_head=True,
                        )  # change for x'(t0)

                    elif self.opts.problem_type == "BVP":
                        plt.arrow(
                            inputs[0, -1],
                            outputs[0, 0],
                            0,
                            delta_c / doutputs[idx_c, 0, 0],
                            ec=GREEN,
                            fc=GREEN,
                            head_width=head_width,
                            head_length=head_length,
                            length_includes_head=True,
                        )  # change for x(t0)

                        plt.arrow(
                            inputs[-1, -1],
                            outputs[-1, 0],
                            0,
                            delta_c / doutputs[idx_c, 0, self.opts.order_s],
                            ec=BLUE,
                            fc=BLUE,
                            head_width=head_width,
                            head_length=head_length,
                            length_includes_head=True,
                        )  # change for x(tf)

                current_loss = 0
                inputs_new = inputs.clone()
                inputs_new[:, 0] += delta_c / doutputs[idx_c, 0, 0]
                outputs_new = self.model(inputs_new)
                outputs_new = outputs_new.detach()
                current_loss += (outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000
                if idx in [1, 2, 3]:
                    plt.scatter(inputs_new[idx_c, -1], outputs_new[idx_c, 0], c=GREEN)  # point at t_c
                    plt.plot(inputs_new[:, -1], outputs_new[:, 0], color=GREEN, linestyle="--", label=legends[1])

                inputs_new = inputs.clone()
                if self.opts.problem_type == "IVP":
                    inputs_new[:, 1] += delta_c / doutputs[idx_c, 0, 1]
                elif self.opts.problem_type == "BVP":
                    inputs_new[:, self.opts.order_s] += delta_c / doutputs[idx_c, 0, self.opts.order_s]

                outputs_new = self.model(inputs_new)
                outputs_new = outputs_new.detach()
                current_loss += (outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000
                if idx in [1, 2, 3]:
                    plt.scatter(inputs_new[idx_c, -1], outputs_new[idx_c, 0], c=BLUE)  # point at t_c
                    plt.plot(inputs_new[:, -1], outputs_new[:, 0], color=BLUE, linestyle=":", label=legends[2])

                inputs_new = inputs.clone()
                if self.opts.problem_type == "IVP" and self.opts.order_s == 2:
                    inputs_new[:, 2] += delta_c / doutputs[idx_c, 0, 2]
                else:
                    loss.append(current_loss)
                    if idx in [1, 2, 3]:
                        plt.xlabel(r"$t$")
                        plt.ylabel(r"$x(t)$")
                        plt.legend(loc="best")
                        plt.show()
                    idx += 1
                    continue

                outputs_new = self.model(inputs_new)
                outputs_new = outputs_new.detach()
                current_loss += (outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000
                loss.append(current_loss)
                if idx in [1, 2, 3]:
                    plt.scatter(inputs_new[idx_c, -1], outputs_new[idx_c, 0])  # point at t_c
                    plt.plot(inputs_new[:, -1], outputs_new[:, 0], linestyle="-.", label=r"modify $x''(t_0)$")
                    plt.xlabel(r"$t$")
                    plt.ylabel(r"$x(t)$")
                    plt.legend(loc="best")
                    plt.show()

            idx += 1

        sorted_indices = np.argsort(loss)
        for i in sorted_indices:
            print(f"{i} \t {loss[i]}")

        print(sum(loss) / len(loss))

    def test_inverted_pendulum(self):
        assert self.opts.problem_type == "BVP", "Not implemented."
        assert self.opts.order_s == 1, "Not implemented."

        loss, l_c = [], []
        for idx, (inputs, labels) in enumerate(tqdm(self.test_dataloader)):
            input_t = inputs[:, -1:]
            input_t.requires_grad_(True)
            inputs = torch.cat([inputs[:, :-1], input_t], dim=1)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            num_outputs = outputs.shape[1]
            doutputs = torch.stack([getGrad(outputs[:, i], inputs) for i in range(num_outputs)], dim=1)

            losses = getLoss(self.criterion, outputs, inputs, labels, self.opts, input_t)
            loss.append(losses["loss"].item())

            inputs = inputs.detach()
            outputs = outputs.detach()
            doutputs = doutputs.detach()

            t_0 = self.opts.t_0
            t_f = self.opts.t_f
            duration = t_f - t_0

            t_c = t_0 + 0.5 * duration
            idx_c = int((t_c - t_0) / duration * (self.opts.num_samples_per_curve - 1))
            delta_c = 0.1

            inputs_new = inputs.clone()
            inputs_new[:, 2] += delta_c / doutputs[idx_c, 1, 2]
            outputs_new = self.model(inputs_new)
            outputs_new = outputs_new.detach()
            l_c.append((outputs_new[idx_c, 1] - outputs[idx_c, 1] - delta_c) ** 2 * 10000)
            if idx in [3]:
                print(idx)
                plt.figure(1)
                plt.subplot(3, 1, 1)
                plt.tight_layout()

                # legends = ["Reference", "Proposed", "Modified"]
                legends = ["参考真值", "微调前", "微调后"]

                plt.plot(inputs[:, -1], labels[:, 0], linestyle="-", color=RED, label=legends[0])
                plt.plot(inputs[:, -1], outputs[:, 0], linestyle="--", color=GREEN, label=legends[1])
                plt.plot(inputs[:, -1], outputs_new[:, 0], linestyle="-.", color=BLUE, label=legends[2])
                plt.xticks([])
                plt.ylabel(r"$\theta(t)$")
                plt.legend(loc="best")

                plt.subplot(3, 1, 2)
                plt.tight_layout()
                plt.plot(inputs[:, -1], labels[:, 1], linestyle="-", color=RED)
                plt.plot(inputs[:, -1], outputs[:, 1], linestyle="--", color=GREEN)
                plt.plot(inputs[:, -1], outputs_new[:, 1], linestyle="-.", color=BLUE)
                plt.xticks([])
                plt.ylabel(r"$x(t)$")

                plt.subplot(3, 1, 3)
                plt.tight_layout()
                plt.plot(inputs[:, -1], labels[:, 2], linestyle="-", color=RED)
                plt.plot(inputs[:, -1], outputs[:, 2], linestyle="--", color=GREEN)
                plt.plot(inputs[:, -1], outputs_new[:, 2], linestyle="-.", color=BLUE)
                plt.xlabel(r"$t$")
                plt.ylabel(r"$F(t)$")

                inputs_numpy = inputs.numpy()
                outputs_numpy = outputs.numpy()

                inputs_new_numpy = inputs_new.numpy()
                outputs_new_numpy = outputs_new.numpy()

                fig, ax = plt.subplots()
                l = 0.3
                x = outputs_numpy[idx_c, 1] - 2 * l * np.sin(outputs_numpy[idx_c, 0])
                y = 2 * l * np.cos(outputs_numpy[idx_c, 0])
                ax.scatter(x, y, c="r", s=250)

                plot_ghost(fig, ax, inputs_numpy[:, -1], outputs_numpy[:, :], "c", legends[1])
                plot_ghost(fig, ax, inputs_new_numpy[:, -1], outputs_new_numpy[:, :], "m", legends[2])
                ax.legend(loc="best")

                # ani = plot_inverted_pendulum_animation_compare(fig, ax, inputs_numpy[:, -1], outputs_numpy[:, :], outputs_new_numpy[:, :], "c", "m")
                # ani0 = plot_inverted_pendulum_animation(fig, ax, inputs_numpy[:, -1], outputs_numpy[:, :], "c")
                # ani1 = plot_inverted_pendulum_animation(fig, ax, inputs_new_numpy[:, -1], outputs_new_numpy[:, :], "m")
                plt.tight_layout()
                plt.show()

        sorted_indices = np.argsort(loss)
        for i in sorted_indices:
            print(f"{i} \t {loss[i]}")
        print(sum(loss) / len(loss))

        sorted_indices = np.argsort(l_c)
        for i in sorted_indices:
            print(f"{i} \t {l_c[i]}")
        print(sum(l_c) / len(l_c))

    def test_poly(self):
        assert self.opts.problem_type == "IVP", "Not implemented."
        assert self.opts.order_s == 2, "Not implemented."
        assert self.opts.num_vars == 1, "Not implemented."
        assert self.opts.ode_func == "odePolyVel", "Not implemented."

        l_c = []
        head_width, head_length = 0.02, 0.05
        for idx, (inputs, labels) in enumerate(tqdm(self.test_dataloader)):
            input_t = inputs[:, -1:]
            input_t.requires_grad_(True)
            inputs = torch.cat([inputs[:, :-1], input_t], dim=1)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            num_outputs = outputs.shape[1]
            doutputs = torch.stack([getGrad(outputs[:, i], inputs) for i in range(num_outputs)], dim=1)

            inputs = inputs.detach()
            outputs = outputs.detach()
            doutputs = doutputs.detach()

            t_0 = self.opts.t_0
            t_f = self.opts.t_f
            duration = t_f - t_0

            t_c = t_0 + 0.5 * duration
            idx_c = int((t_c - t_0) / duration * (self.opts.num_samples_per_curve - 1))
            delta_c = 0.2

            inputs_new = inputs.clone()
            inputs_new[:, 2] += delta_c / doutputs[idx_c, 0, 2]
            outputs_new = self.model(inputs_new)
            outputs_new = outputs_new.detach()

            l_c.append((outputs_new[idx_c, 0] - outputs[idx_c, 0] - delta_c) ** 2 * 10000)
            if idx in [91, 29, 41, 11, 47]:
                print(idx)
                plt.figure(1)
                plt.tight_layout()
                legends = ["微调前", "微调后"]
                plt.plot(inputs[:, -1], outputs[:, 0], linestyle="-", color=RED, label=legends[0])
                plt.scatter(inputs[idx_c, -1], outputs[idx_c, 0], c=RED)  # point at t_c
                plt.arrow(
                    inputs[idx_c, -1],
                    outputs[idx_c, 0],
                    0,
                    delta_c,
                    ec=RED,
                    fc=RED,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=True,
                )  # arrow for delta_c
                plt.plot(inputs[:, -1], outputs_new[:, 0], linestyle="--", color=GREEN, label=legends[1])
                plt.scatter(inputs_new[idx_c, -1], outputs_new[idx_c, 0], c=GREEN)  # point at t_c
                plt.xlabel(r"$t\ (\rm{s})$")
                plt.ylabel(r"$p(t)\ (\rm{m})$")
                plt.legend(loc="best")

                plt.figure(2)
                plt.subplot(2, 1, 1)
                plt.tight_layout()
                plt.plot(inputs[:, -1], outputs[:, 1], linestyle="-", color=RED)
                plt.plot(inputs[:, -1], outputs_new[:, 1], linestyle="--", color=GREEN)
                plt.xticks([])
                plt.ylabel(r"$v(t)\ (\rm{m} \cdot \rm{s}^{-1})$")

                plt.subplot(2, 1, 2)
                plt.tight_layout()
                plt.plot(inputs[:, -1], outputs[:, 2], linestyle="-", color=RED)
                plt.plot(inputs[:, -1], outputs_new[:, 2], linestyle="--", color=GREEN)
                plt.xlabel(r"$t\ (\rm{s})$")
                plt.ylabel(r"$a(t)\ (\rm{m} \cdot \rm{s}^{-2})$")
                plt.show()

        sorted_indices = np.argsort(l_c)
        for i in sorted_indices:
            print(f"{i} \t {l_c[i]}")
        print(sum(l_c) / len(l_c))

    def test_high_derivative(self):
        dataiter = iter(self.test_dataloader)
        inputs, _ = next(dataiter)

        input_t = inputs[:, -1:]
        input_t.requires_grad_(True)
        inputs = torch.cat([inputs[:, :-1], input_t], dim=1)

        outputs = self.model(inputs)
        num_outputs = outputs.shape[1]

        doutputs = torch.stack([getGrad(outputs[:, i], input_t) for i in range(num_outputs)], dim=1)  # [batch_size, num_outputs, 1]

        inputs = inputs.detach()
        outputs = outputs.detach()
        doutputs = doutputs.detach()

        plt.plot(inputs[:, -1], outputs[:, 0], color=RED, label=r"$x(t)$")
        plt.plot(inputs[:, -1], outputs[:, 1], color=GREEN, label=r"$x'(t)$")
        plt.plot(inputs[:, -1], doutputs[:, 0, -1], color=RED, linestyle="--", label=r"$\frac{dx(t)}{dt}$")
        plt.plot(inputs[:, -1], doutputs[:, 1, -1], color=GREEN, linestyle="--", label=r"$\frac{dx'(t)}{dt}$")

        # plt.plot(inputs[:, -1], outputs[:, 2], color="b", label="x''(t)")
        # plt.plot(inputs[:, -1], outputs[:, 3], color="m", label="x''(t)")
        # plt.plot(inputs[:, -1], doutputs[:, 2, -1], color="b", linestyle="--", label="dx''(t)")

        # plt.plot(inputs[:, -1], outputs[:, 4], color="r", label="x''(t)")
        # plt.plot(inputs[:, -1], doutputs[:, 3, -1], color="m", linestyle="--", label="dx''(t)")

        # plt.plot(inputs[:, -1], outputs[:, 5], color="g", label="x''(t)")
        # plt.plot(inputs[:, -1], doutputs[:, 4, -1], color="r", linestyle="--", label="dx''(t)")

        # plt.plot(inputs[:, -1], doutputs[:, 5, -1], color="g", linestyle="--", label="dx''(t)")

        # plt.plot(inputs[:, -1], outputs[:, 1], color="r", label="x(t)")
        # plt.plot(inputs[:, -1], doutputs[:, 0, -1], color="g", label="x'(t)")
        # plt.plot(inputs[:, -1], doutputs[:, self.opts.num_variables, -1], color="b", label="x''(t)")

        plt.legend(loc="best")
        plt.show()

    def test_grad_time(self):
        assert self.opts.problem_type in ["BVP"], "Not implemented."
        assert self.opts.order_s == 1, "Not implemented."
        assert self.opts.ode_func == "odeVdp", "Not implemented."

        t_0 = self.opts.t_0
        t_f = self.opts.t_f
        duration = t_f - t_0

        t_c = t_0 + 0.5 * duration
        idx_c = int((t_c - t_0) / duration * (self.opts.num_samples_per_curve - 1))
        delta_c = 0.5

        ours_forward_time, ours_finetune_time = [], []
        scip_forward_time, scip_finetune_time = [], []
        for inputs, _ in tqdm(self.test_dataloader):
            self.fa = inputs[0, 0].item()
            self.fb = inputs[0, 1].item()
            inputs.requires_grad_(True)

            # ======== proposed start ========
            ours_forward_start = timer()
            outputs = self.model(inputs)
            ours_forward_end = timer()
            ours_forward_time.append(ours_forward_end - ours_forward_start)

            ours_finetune_start = timer()
            if self.opts.one_output:
                doutputs = getGrad(outputs, inputs)
                inputs = inputs.detach()
                inputs[:, 0] += delta_c / doutputs[idx_c, 0]
                outputs = self.model(inputs)
            else:
                num_outputs = outputs.shape[1]
                doutputs = torch.stack([getGrad(outputs[:, i], inputs) for i in range(num_outputs)], dim=1)
                inputs = inputs.detach()
                inputs[:, 0] += delta_c / doutputs[idx_c, 0, 0]
                outputs = self.model(inputs)
            ours_finetune_end = timer()
            ours_finetune_time.append(ours_finetune_end - ours_finetune_start)
            # ======== proposed end ========

            # ======== solver start ========
            scip_forward_start = timer()
            tSol = np.linspace(t_0, t_f, self.opts.num_samples_per_curve)

            t = np.linspace(t_0, t_f, self.opts.num_samples_per_curve)
            x = np.zeros((2, t.shape[0]))
            res = solve_bvp(self.odeVdp, self.bc, t, x)
            xSol = res.sol(tSol)[0]
            scip_forward_end = timer()
            scip_forward_time.append(scip_forward_end - scip_forward_start)

            scip_finetune_start = timer()
            self.fb = xSol[idx_c] + delta_c
            t = np.linspace(t_0, t_c, self.opts.num_samples_per_curve)
            x = np.zeros((2, t.shape[0]))
            res = solve_bvp(self.odeVdp, self.bc, t, x)
            xSol = res.sol(tSol)[0]
            scip_finetune_end = timer()
            scip_finetune_time.append(scip_finetune_end - scip_finetune_start)
            # ======== solver end ========

        ours_forward_time = self.calcAvg(ours_forward_time)
        ours_finetune_time = self.calcAvg(ours_finetune_time)
        our_total_time = ours_forward_time + ours_finetune_time
        scip_forward_time = self.calcAvg(scip_forward_time)
        scip_finetune_time = self.calcAvg(scip_finetune_time)
        scip_total_time = scip_forward_time + scip_finetune_time
        print(f"ours: forward: {ours_forward_time}, finetune: {ours_finetune_time}, total: {our_total_time}")
        print(f"scip: forward: {scip_forward_time}, finetune: {scip_finetune_time}, total: {scip_total_time}")

    def calcAvg(self, time_list):
        s = sum(time_list) - max(time_list) - min(time_list)
        n = len(time_list) - 2
        return s / n

    def odeExpBvp(self, t, x):
        x0, x1 = x
        dxdt = np.array([x1, -x1 + 2 * x0])
        return dxdt

    def odeVdp(self, t, x):
        x0, x1 = x
        dxdt = np.array([x1, (1 - x0**2) * x1 - x0])
        return dxdt

    def bc(self, xa, xb):
        return np.array([xa[0] - self.fa, xb[0] - self.fb])
