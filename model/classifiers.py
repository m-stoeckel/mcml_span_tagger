import torch
from torch import nn as nn


class LinearMultiHeadClassifier(nn.Module):
    """
    Multi-head classifier module using multiple linear layers in a ModuleList.
    May be faster on CPU than the alternative Conv1D implementation.

    High batch sizes minimize runtime on GPUs.
    """

    def __init__(self, input_size, num_heads, output_size=1):
        super(LinearMultiHeadClassifier, self).__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(input_size, output_size) for _ in range(num_heads)]
        )

    def forward(self, x) -> torch.Tensor:
        x = torch.cat(
            [classifier.forward(x) for classifier in self.classifiers], dim=-1
        )
        return x


class CudaStreamMultiHeadClassifier(LinearMultiHeadClassifier):
    """
    Same as the LinearMultiHeadClassifier but utilizes a pool of CUDA stream for asynchronous
    computation. However, it is always slower than the LinearMultiHeadClassifier.
    """

    def __init__(self, input_size, num_heads, num_streams=2):
        super(CudaStreamMultiHeadClassifier, self).__init__(input_size, num_heads)
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]

    def forward(self, x) -> torch.Tensor:
        x_temp = []

        for idx, classifier in list(enumerate(self.classifiers)):
            with torch.cuda.stream(self.streams[idx % self.num_streams]):
                x_temp.append(classifier.forward(x))
        torch.cuda.synchronize()

        x = torch.cat(x_temp, dim=-1)
        return x


class ConvolutionalMultiHeadClassifier(nn.Module):
    """
    Multi-head classifier module using a single 1d convolutional layer with kernel size 1.
    Significantly faster than the multi-head linear classifier when run on a GPU.

    Reducing the kernel size to 1, effectively turns a convolutional layer into a linear layer.
    Each input dimension gets its own channel and thus its own weight (as in a regular linear layer).
    The groups argument is necessary to enforce different weights for each output channel.

    Medium batch sizes (128, 256) minimize runtime on GPUs.
    """

    def __init__(self, input_size, num_heads):
        super(ConvolutionalMultiHeadClassifier, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.conv = nn.Conv1d(input_size * num_heads, num_heads, 1, groups=num_heads)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size(0)

        # Reshape the input to match the number of input channels on the second dimension.
        x = x.reshape(-1, self.input_size).unsqueeze(-1).repeat(1, self.num_heads, 1)

        # Run the convolution
        x = self.conv(x)

        # Return the output back to the expected shape
        x = x.squeeze().reshape(batch_size, -1, self.num_heads)
        return x