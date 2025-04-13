import torch


class LinearWithResidual(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        assert in_features == out_features, "in_features and out_features should be equal"
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x) + x


class MLPR(torch.nn.Module):
    def __init__(self, num_inputs, num_neurals=32, num_layers=4, num_outputs=1):
        super().__init__()

        module_list = torch.nn.ModuleList()

        # input layer
        module_list.append(
            torch.nn.Sequential(
                torch.nn.Linear(num_inputs, num_neurals),
                torch.nn.Tanh(),
            )
        )

        # hidden layers
        for _ in range(num_layers - 1):
            module_list.append(
                torch.nn.Sequential(
                    LinearWithResidual(num_neurals, num_neurals),
                    torch.nn.Tanh(),
                )
            )

        # output layer
        module_list.append(torch.nn.Linear(num_neurals, num_outputs))

        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)
