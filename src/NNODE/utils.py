import random

import numpy as np
import ode_func
import torch
from loguru import logger
from matplotlib import pyplot as plt


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"using seed: {seed}")


def getGrad(outputs, inputs):
    (grad,) = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )
    return grad


def getLoss(criterion, outputs, inputs, labels, opts, input_t=None):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    t_0, t_f = opts.t_0, opts.t_f

    # losses initialization
    losses = {}  # loss dict
    losses["l_f"] = torch.tensor(0.0, device=device)  # ode constraint
    losses["l_b"] = torch.tensor(0.0, device=device)  # boundary conditions
    losses["l_r"] = torch.tensor(0.0, device=device)  # regression

    if opts.has_regression_loss:
        if opts.one_output:
            losses["l_r"] += criterion(outputs, labels[:, : opts.num_vars])
        else:
            losses["l_r"] += criterion(outputs, labels)

    # model one output, TODO: not updated
    if opts.one_output:
        assert opts.order_s == 1, "Not implemented."

        doutputs = getGrad(outputs[:, 0], input_t)
        ddoutputs = getGrad(doutputs[:, 0], input_t)

        cat_outputs = torch.cat([outputs, doutputs], dim=1)
        cat_doutputs = torch.cat([doutputs, ddoutputs], dim=1)
        dcat_outputs = getattr(ode_func, opts.ode_func + "Tensor")(cat_outputs, opts)

        # derivative constraint
        losses["l_f"] += criterion(cat_doutputs, dcat_outputs)

        if opts.problem_type == "IVP":
            # fixed state at t0
            initial_idx = inputs[:, -1] == t_0
            if initial_idx.any():
                losses["l_b"] += criterion(cat_outputs[initial_idx], inputs[initial_idx, :-1])

        elif opts.problem_type == "BVP":
            # fixed state at t0
            initial_idx = inputs[:, -1] == t_0
            if initial_idx.any():
                losses["l_b"] += criterion(outputs[initial_idx, : opts.order_s * opts.num_vars], inputs[initial_idx, : opts.order_s * opts.num_vars])

            # fixed state at tf
            terminal_idx = inputs[:, -1] == t_f
            if terminal_idx.any():
                losses["l_b"] += criterion(outputs[terminal_idx, : opts.order_s * opts.num_vars], inputs[terminal_idx, opts.order_s * opts.num_vars : -1])

    # get derivatives
    elif not opts.only_regression_loss and input_t is not None:
        num_outputs = outputs.shape[1]

        doutputs = torch.stack([getGrad(outputs[:, i], input_t) for i in range(num_outputs)], dim=1)  # [batch_size, num_outputs, 1]
        doutputs.squeeze_()  # [batch_size, num_outputs]
        if opts.ode_func == "odeVdpMu":
            input_mu = inputs[:, -2]
            dxdt = getattr(ode_func, opts.ode_func + "Tensor")(outputs, opts, input_mu)
        else:
            dxdt = getattr(ode_func, opts.ode_func + "Tensor")(outputs, opts)

        # derivative constraint
        losses["l_f"] += criterion(doutputs, dxdt)

        if opts.problem_type == "IVP":
            # fixed state at t0
            initial_idx = inputs[:, -1] == t_0
            if initial_idx.any():
                losses["l_b"] += criterion(outputs[initial_idx], inputs[initial_idx, : 2 * opts.num_vars * opts.order_s])

        elif opts.problem_type == "BVP":
            # fixed state at t0
            initial_idx = inputs[:, -1] == t_0
            if initial_idx.any():
                losses["l_b"] += criterion(outputs[initial_idx, : opts.order_s * opts.num_vars], inputs[initial_idx, : opts.order_s * opts.num_vars])

            # fixed state at tf
            terminal_idx = inputs[:, -1] == t_f
            if terminal_idx.any():
                losses["l_b"] += criterion(outputs[terminal_idx, : opts.order_s * opts.num_vars], inputs[terminal_idx, opts.order_s * opts.num_vars : 2 * opts.order_s * opts.num_vars])

    losses["loss"] = opts.lambda_f * losses["l_f"] + opts.lambda_b * losses["l_b"] + opts.lambda_r * losses["l_r"]
    return losses


def plot_inverted_pendulum(t, xt, l=0.3):
    plt.figure(figsize=(10, 8))

    x = xt[:, 1]
    y = np.zeros_like(x)
    theta = xt[:, 0]
    pendulum_x = x - 2 * l * np.sin(theta)
    pendulum_y = y + 2 * l * np.cos(theta)

    for i in range(len(t)):
        plt.plot([x[i], pendulum_x[i]], [y[i], pendulum_y[i]], "b-", linewidth=2)
        plt.plot(x[i], y[i], "ro")

    # plt.xlim(min(x) - 2 * l, max(x) + 2 * l)
    # plt.ylim(-2 * l, 2 * l)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_inverted_pendulum_animation(fig, ax, t, xt, c="r", l=0.3):
    from matplotlib.animation import FuncAnimation

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(min(xt[:, 1]) - 2 * l * 1.2, max(xt[:, 1]) + 2 * l * 1.2)
    ax.set_ylim(-2 * l * 1.2, 2 * l * 1.2)
    ax.set_aspect("equal")
    ax.grid(True)

    (line,) = ax.plot([], [], "b-", linewidth=2)
    (point,) = ax.plot([], [], "ro")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        x = xt[i, 1]
        y = 0
        theta = xt[i, 0]
        pendulum_x = x - 2 * l * np.sin(theta)
        pendulum_y = y + 2 * l * np.cos(theta)
        line.set_data([x, pendulum_x], [y, pendulum_y])
        point.set_data([x], [y])
        return line, point

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(t), blit=True, interval=10, repeat=True)
    return ani


def plot_inverted_pendulum_animation_compare(fig, ax, t, xt0, xt1, c0="r", c1="b", l=0.3):
    from matplotlib.animation import FuncAnimation

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(min(min(xt0[:, 1]), min(xt1[:, 1])) - 2 * l * 1.2, max(max(xt0[:, 1]), max(xt1[:, 1])) + 2 * l * 1.2)
    ax.set_ylim(-2 * l * 1.2, 2 * l * 1.2)
    ax.set_aspect("equal")
    ax.grid(True)

    (line0,) = ax.plot([], [], c0 + "-", linewidth=2)
    (line1,) = ax.plot([], [], c1 + "-", linewidth=2)

    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        return line0, line1

    def animate(i):
        x = xt0[i, 1]
        y = 0
        theta = xt0[i, 0]
        pendulum_x = x - 2 * l * np.sin(theta)
        pendulum_y = y + 2 * l * np.cos(theta)
        line0.set_data([x, pendulum_x], [y, pendulum_y])

        x = xt1[i, 1]
        y = 0
        theta = xt1[i, 0]
        pendulum_x = x - 2 * l * np.sin(theta)
        pendulum_y = y + 2 * l * np.cos(theta)
        line1.set_data([x, pendulum_x], [y, pendulum_y])
        return line0, line1

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(t), blit=True, interval=10, repeat=True)
    # ani.save("animation.mp4", dpi=300, writer="ffmpeg")
    return ani


def plot_ghost(fig, ax, t, xt, c="r", label="", l=0.3):
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(min(xt[:, 1]) - 2 * l * 1.1, max(xt[:, 1]) + 2 * l * 1.1)
    ax.set_ylim(-2 * l * 1.1, 2 * l * 1.1)
    ax.set_aspect("equal")
    ax.grid(True)

    alpha_step = 0.8 / len(t)
    for i in range(len(t)):
        x = xt[i, 1]
        y = 0
        theta = xt[i, 0]
        pendulum_x = x - 2 * l * np.sin(theta)
        pendulum_y = y + 2 * l * np.cos(theta)
        ax.plot([x, pendulum_x], [y, pendulum_y], c=c, linewidth=2, alpha=alpha_step * i)

    x = xt[-1, 1]
    y = 0
    theta = xt[-1, 0]
    pendulum_x = x - 2 * l * np.sin(theta)
    pendulum_y = y + 2 * l * np.cos(theta)
    ax.plot([x, pendulum_x], [y, pendulum_y], c=c, linewidth=2, alpha=1, label=label)
