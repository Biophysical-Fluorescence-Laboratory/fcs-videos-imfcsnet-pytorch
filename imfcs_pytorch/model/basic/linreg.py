import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(
        self,
        output_neurons: int,
    ):
        super(LinearRegressionModel, self).__init__()

        # This MLP layer converts the feature-space into the output nodes that are required.
        # In the case of the original D-regression network, this would be 1
        self.fc = nn.Linear(2, output_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Model expects inputs with 2 features, the mean and the stdev.
        # Prepare the inputs in the shape (BATCH_SIZE, 2)
        with torch.no_grad():
            stdev_batch, mean_batch = torch.std_mean(x, dim=(1, 2, 3, 4))
            x = torch.stack([stdev_batch, mean_batch], dim=1)

        x = self.fc(x)
        return x


def build_linear_regression(
    output_neurons: int,
) -> LinearRegressionModel:
    return LinearRegressionModel(output_neurons=output_neurons)
