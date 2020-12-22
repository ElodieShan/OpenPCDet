import spconv
import torch.nn as nn
import torch

class MappingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1):
        super(MappingBlock, self).__init__()

        self.conv_map = nn.Sequential(
                            nn.Conv1d(
                                input_channels, output_channels,
                                kernel_size=kernel_size, stride=stride
                            ),
                            nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            nn.Conv1d(
                                output_channels, output_channels,
                                kernel_size=1, 
                            ),
                    )

    def forward(self, x):
        out = self.conv_map(x)
        return out
