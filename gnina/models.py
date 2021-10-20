from collections import OrderedDict

import torch
from torch import nn

from typing import Tuple


class Default2017(nn.Module):
    """
    GNINA default2017 model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        self.input_dims = input_dims

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # unit1
                    ("unit1_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit1_conv",
                        nn.Conv3d(
                            in_channels=input_dims[0],
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit1_func", nn.ReLU()),
                    # unit2
                    ("unit2_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit2_conv",
                        nn.Conv3d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit2_func", nn.ReLU()),
                    # unit3
                    ("unit3_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit3_conv",
                        nn.Conv3d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit3_func", nn.ReLU()),
                ]
            )
        )

        self.features_out_size = (
            input_dims[1] // 8 * input_dims[2] // 8 * input_dims[3] // 8 * 128
        )

        self.pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

        self.affinity = nn.Sequential(
            OrderedDict(
                [
                    (
                        "affinity_output",
                        nn.Linear(in_features=self.features_out_size, out_features=1),
                    )
                ]
            )
        )

        # TODO: Check that Caffe's Xavier is xavier_uniform_ (not xavier_normal_)
        # Xavier initialization for convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Notes
        -----
        The pose score is the raw output of a linear layer. Softmax is applied with the
        loss.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        affinity = self.affinity(x)

        return pose_raw, affinity


class Default2018(nn.Module):
    """
    GNINA default2017 model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        self.input_dims = input_dims

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # unit1
                    ("unit1_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
                    (
                        "unit1_conv",
                        nn.Conv3d(
                            in_channels=input_dims[0],
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit1_func", nn.ReLU()),
                    # unit2
                    (
                        "unit2_conv",
                        nn.Conv3d(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                    ),
                    ("unit2_func", nn.ReLU()),
                    # unit3
                    ("unit3_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
                    (
                        "unit3_conv",
                        nn.Conv3d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit3_func", nn.ReLU()),
                    # unit4
                    (
                        "unit4_conv",
                        nn.Conv3d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                    ),
                    ("unit4_func", nn.ReLU()),
                    # unit5
                    ("unit5_pool", nn.AvgPool3d(kernel_size=2, stride=2)),
                    (
                        "unit5_conv",
                        nn.Conv3d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit5_func", nn.ReLU()),
                ]
            )
        )

        self.features_out_size = (
            input_dims[1] // 8 * input_dims[2] // 8 * input_dims[3] // 8 * 128
        )

        self.pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

        self.affinity = nn.Sequential(
            OrderedDict(
                [
                    (
                        "affinity_output",
                        nn.Linear(in_features=self.features_out_size, out_features=1),
                    )
                ]
            )
        )

        # TODO: Check that Caffe's Xavier is xavier_uniform_ (not xavier_normal_)
        # Xavier initialization for convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Notes
        -----
        The pose score is the raw output of a linear layer. Softmax is applied with the
        loss.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        affinity = self.affinity(x)

        return pose_raw, affinity
