import torch
from torch import nn


class HistogramOfFeaturesModel(nn.Module):
    def __init__(
        self,
        histogram_bins: int,
        output_neurons: int,
    ):
        super(HistogramOfFeaturesModel, self).__init__()

        self.histogram_bins = histogram_bins
        self.fc = nn.Linear((histogram_bins * 2 + 1), output_neurons)

    @staticmethod
    def histogram(x: torch.Tensor, bins: int, density: bool = False):
        """Workaround for histogram()'s inability to work on GPU tensors.

        See https://github.com/pytorch/pytorch/issues/69519#issuecomment-1183866843"""
        # Extract the device of the input, ensures that the boundaries are on the same device.
        device = x.get_device()

        min, max = x.min(), x.max()
        counts = torch.histc(x, bins, min=min, max=max)

        # If density = True, compute the density instead of counts
        if density:
            counts = counts / torch.sum(counts)

        boundaries = torch.linspace(min, max, bins + 1, device=device)
        return counts, boundaries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten inputs batch-wise.
        with torch.no_grad():
            x = torch.flatten(x, start_dim=1, end_dim=-1)

            # Transform the inputs into histograms.
            x = torch.stack(
                [
                    torch.cat(self.histogram(b, bins=self.histogram_bins, density=True))
                    for b in x
                ],
                dim=0,
            )

        x = self.fc(x)
        return x


def build_histogram_of_features(
    histogram_bins: int,
    output_neurons: int,
) -> HistogramOfFeaturesModel:
    return HistogramOfFeaturesModel(
        histogram_bins=histogram_bins, output_neurons=output_neurons
    )
