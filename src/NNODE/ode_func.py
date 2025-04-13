import numpy as np
import torch
from scipy.special import comb


def odeFreeFall(x, t):
    x0, x1 = x
    dxdt = np.array([x1, -9.8])
    return dxdt


def odeFreeFallTensor(x, opts):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, 0] = x[:, 1]
    dxdt[:, 1] = -9.8
    return dxdt


def odePoly(x, t):
    dxdt = np.zeros_like(x)
    dxdt[:-1] = x[1:]
    return dxdt


def odePolyTensor(x, opts):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, :-1] = x[:, 1:]
    return dxdt


def odePolyVel(x, t):
    dxdt = np.zeros_like(x)
    mu, v_max = 1.0, 5.0
    dxdt[:-1] = x[1:]
    dxdt[-1] = (mu * x[2] * (v_max**2 - x[1] ** 2) + 2 * mu * x[2] * x[1] ** 2) / (v_max**2 - x[1] ** 2) ** 2
    return dxdt


def odePolyVelTensor(x, opts):
    mu, v_max = 1.0, 5.0
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, :-1] = x[:, 1:]
    dxdt[:, -1] = (mu * x[:, 2] * (v_max**2 - x[:, 1] ** 2) + 2 * mu * x[:, 2] * x[:, 1] ** 2) / (v_max**2 - x[:, 1] ** 2) ** 2
    return dxdt


def odeInvertedPendulum(x, t):
    """
    x0 = theta, x1 = x, x2 = F, x3 = \dot{theta}, x4 = \dot{x}, x5 = \dot{F}
    """
    J = 0.006
    M = 0.5
    L = 0.3
    m = 0.2
    g = 9.81

    dxdt = np.zeros_like(x)
    dxdt[:3] = x[3:]

    A = np.array([[J + m * L**2, -m * L * np.cos(x[0])], [-m * L * np.cos(x[0]), M + m]])
    b = np.array([[m * g * L * np.sin(x[0])], [x[2] - m * L * np.sin(x[0]) * x[3] ** 2]])

    p = np.linalg.solve(A, b)
    dxdt[3] = p[0]
    dxdt[4] = p[1]
    return dxdt


def odeInvertedPendulumTensor(x, opts):
    """
    x0 = theta, x1 = x, x2 = F, x3 = \dot{theta}, x4 = \dot{x}, x5 = \dot{F}
    """
    J = 0.006
    M = 0.5
    L = 0.3
    m = 0.2
    g = 9.81

    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, :3] = x[:, 3:]

    A_11 = J + m * L**2 * torch.ones_like(x[:, 0], device=device)
    A_12 = -m * L * torch.cos(x[:, 0])
    A_21 = -m * L * torch.cos(x[:, 0])
    A_22 = M + m * torch.ones_like(x[:, 0], device=device)
    A = torch.stack([torch.stack([A_11, A_12], dim=1), torch.stack([A_21, A_22], dim=1)], dim=1)

    b_1 = m * g * L * torch.sin(x[:, 0])
    b_2 = x[:, 2] - m * L * torch.sin(x[:, 0]) * x[:, 3] ** 2
    b = torch.stack([b_1, b_2], dim=1)

    p = torch.linalg.solve(A, b)
    dxdt[:, 3] = p[:, 0]
    dxdt[:, 4] = p[:, 1]
    return dxdt


def odeVdp(x, t):
    x0, x1 = x
    dxdt = np.array([x1, (1 - x0**2) * x1 - x0])
    return dxdt


def odeVdpTensor(x, opts):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, 0] = x[:, 1]
    dxdt[:, 1] = (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
    return dxdt


def odeVdpMu(x, t, mu):
    x0, x1 = x
    dxdt = np.array([x1, mu * (1 - x0**2) * x1 - x0])
    return dxdt


def odeVdpMuTensor(x, opts, mu):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, 0] = x[:, 1]
    dxdt[:, 1] = mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
    return dxdt


def odeExp(x, t):
    x0, x1 = x
    dxdt = np.array([x1, -x1 + 2 * x0])
    return dxdt


def odeExpTensor(x, opts):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, 0] = x[:, 1]
    dxdt[:, 1] = -x[:, 1] + 2 * x[:, 0]
    return dxdt


def odeInvRatio(x, t):
    x0, x1 = x
    dxdt = np.array([x1, x0 * x0 * x0 - x0 * x1])
    return dxdt


def odeInvRatioTensor(x, opts):
    device = torch.device("cpu" if opts.no_cuda or not torch.cuda.is_available() else "cuda")
    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, 0] = x[:, 1]
    dxdt[:, 1] = x[:, 0] * x[:, 0] * x[:, 0] - x[:, 0] * x[:, 1]
    return dxdt


def quatHessian(r, i):
    if i == 1:
        # [r(1), 0, r(3), -r(2); 0, r(1), r(2), r(3); r(3), r(2), -r(1), 0; -r(2), r(3), 0, -r(1)]
        hess = np.array(
            [
                [r[0], 0, r[2], -r[1]],
                [0, r[0], r[1], r[2]],
                [r[2], r[1], -r[0], 0],
                [-r[1], r[2], 0, -r[0]],
            ]
        )
    elif i == 2:
        #  [r(2), -r(3), 0, r(1); -r(3), -r(2), r(1), 0; 0, r(1), r(2), r(3); r(1), 0, r(3), -r(2)]
        hess = np.array(
            [
                [r[1], -r[2], 0, r[0]],
                [-r[2], -r[1], r[0], 0],
                [0, r[0], r[1], r[2]],
                [r[0], 0, r[2], -r[1]],
            ]
        )
    elif i == 3:
        # [r(3), r(2), -r(1), 0; r(2), -r(3), 0, r(1); -r(1), 0, -r(3), r(2); 0, r(1), r(2), r(3)];
        hess = np.array(
            [
                [r[2], r[1], -r[0], 0],
                [r[1], -r[2], 0, r[0]],
                [-r[0], 0, -r[2], r[1]],
                [0, r[0], r[1], r[2]],
            ]
        )

    return hess


def quatHessianTensor(r, i, device):
    if i == 1:
        # [r(1), 0, r(3), -r(2); 0, r(1), r(2), r(3); r(3), r(2), -r(1), 0; -r(2), r(3), 0, -r(1)]
        hess = torch.tensor(
            [
                [r[0], 0, r[2], -r[1]],
                [0, r[0], r[1], r[2]],
                [r[2], r[1], -r[0], 0],
                [-r[1], r[2], 0, -r[0]],
            ],
            device=device,
        )
    elif i == 2:
        #  [r(2), -r(3), 0, r(1); -r(3), -r(2), r(1), 0; 0, r(1), r(2), r(3); r(1), 0, r(3), -r(2)]
        hess = torch.tensor(
            [
                [r[1], -r[2], 0, r[0]],
                [-r[2], -r[1], r[0], 0],
                [0, r[0], r[1], r[2]],
                [r[0], 0, r[2], -r[1]],
            ],
            device=device,
        )
    elif i == 3:
        # [r(3), r(2), -r(1), 0; r(2), -r(3), 0, r(1); -r(1), 0, -r(3), r(2); 0, r(1), r(2), r(3)];
        hess = torch.tensor(
            [
                [r[2], r[1], -r[0], 0],
                [r[1], -r[2], 0, r[0]],
                [-r[0], 0, -r[2], r[1]],
                [0, r[0], r[1], r[2]],
            ],
            device=device,
        )

    return hess


def odeQuat(x, t):
    d = 2
    r = np.array([0.909140044672029, 0.404053348250552, 0.101021141060964])
    config = np.eye(3)
    N = config.shape[1]
    b = np.zeros((4, 1))
    Jaco = np.zeros((4, 4))
    q = np.expand_dims(x[:4], axis=1)

    for i in range(1, N + 1):
        for j in range(1, 3 + 1):
            H = r[i - 1] * quatHessian(config[:, i - 1], j)
            kk = 0
            for k in range(1, d + 1):
                # qk = x(4*k+1:4*(k+1))
                qk = x[4 * k + 1 - 1 : 4 * (k + 1) - 1 + 1]

                # q2dk = x(end-4*k + 1:end-4*(k-1));
                # in reverse way, it has the same index between python and matlab
                if k == 1:
                    q2dk = x[-1 - 4 * k + 1 :]
                else:
                    q2dk = x[-1 - 4 * k + 1 : -1 - 4 * (k - 1) + 1]

                if k == d:
                    kk = kk + comb(2 * d, k) * qk.T @ H @ q2dk
                else:
                    kk = kk + 2 * comb(2 * d, k) * qk.T @ H @ q2dk

            b = b + kk * H @ q
            Jaco = Jaco + 2 * H @ (q @ q.T) @ H

    dxdt = np.zeros((8 * d, 1))
    dxdt[: -1 - 4 + 1] = np.expand_dims(x[4:], axis=1)
    dxdt[-1 - 3 :] = -np.linalg.solve(Jaco, b)
    dxdt = np.squeeze(dxdt, axis=1)

    return dxdt


def odeQuatTensor(x, device):
    batch_size = x.shape[0]

    d = 2
    r = torch.tensor([0.909140044672029, 0.404053348250552, 0.101021141060964], device=device)
    config = torch.eye(3, device=device)
    N = config.shape[1]
    b = torch.zeros((batch_size, 4, 1), device=device)
    Jaco = torch.zeros((batch_size, 4, 4), device=device)
    q = torch.unsqueeze(x[:, :4], dim=2)

    for i in range(1, N + 1):
        for j in range(1, 3 + 1):
            H = r[i - 1] * quatHessianTensor(config[:, i - 1], j, device)
            kk = 0
            for k in range(1, d + 1):
                # qk = x(4*k+1:4*(k+1))
                qk = torch.unsqueeze(x[:, 4 * k + 1 - 1 : 4 * (k + 1) - 1 + 1], dim=2)

                # q2dk = x(end-4*k + 1:end-4*(k-1));
                # in reverse way, it has the same index between python and matlab
                if k == 1:
                    q2dk = torch.unsqueeze(x[:, -1 - 4 * k + 1 :], dim=2)
                else:
                    q2dk = torch.unsqueeze(x[:, -1 - 4 * k + 1 : -1 - 4 * (k - 1) + 1], dim=2)

                if k == d:
                    kk = kk + comb(2 * d, k) * qk.permute(0, 2, 1) @ H @ q2dk
                else:
                    kk = kk + 2 * comb(2 * d, k) * qk.permute(0, 2, 1) @ H @ q2dk

            b = b + kk * H @ q
            Jaco = Jaco + 2 * H @ (q @ q.permute(0, 2, 1)) @ H

    dxdt = torch.zeros_like(x, device=device)
    dxdt[:, : -1 - 4 + 1] = x[:, 4:]
    dxdt[:, -1 - 3 :] = -torch.linalg.solve(Jaco, b).squeeze(dim=2)

    return dxdt
