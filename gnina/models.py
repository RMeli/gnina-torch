from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch import nn


class Default2017(nn.Module):
    """
    GNINA default2017 model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architectre was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2017.model
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

    Notes
    -----
    This architectre was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2018.model
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


class DenseBlock(nn.Module):
    """
    DenseBlock for Dense model.

    Parameters
    ----------
    in_features: int
        Input features for the first layer
    num_block_features: int
        Number of output features (channels) for the convolutional layers
    num_block_convs: int
        Number of convolutions
    tag: Union[int, str]
        Tag identifying the DenseBlock

    Notes
    -----
    The total number of output features corresponds tothe input features concatenated
    together with all subsequent :code:`num_block_features` produced by the
    convolutional layers (:code:`num_block_convs` times).
    """

    def __init__(
        self,
        in_features: int,
        num_block_features: int = 16,
        num_block_convs: int = 4,
        tag: Union[int, str] = "",
    ) -> None:

        super().__init__()

        self.blocks = nn.ModuleList()

        self.in_features = in_features
        self.num_block_features = num_block_features
        self.num_block_convs = num_block_convs

        in_features_layer = in_features
        for idx in range(num_block_convs):
            block: OrderedDict[str, nn.Module] = OrderedDict()
            block[f"data_enc_level{tag}_batchnorm_conv{idx}"] = nn.BatchNorm3d(
                in_features_layer,
                affine=True,  # Same effect as "Scale" layer in Caffe
            )
            block[f"data_enc_level{tag}_conv{idx}"] = nn.Conv3d(
                in_channels=in_features_layer,
                out_channels=num_block_features,
                kernel_size=3,
                padding=1,
            )
            block[f"data_enc_level{tag}_conv{idx}_relu"] = nn.ReLU()

            self.blocks.append(nn.Sequential(block))

            # The next layer takes all previous features as input
            in_features_layer += num_block_features

    def out_features(self) -> int:
        return self.in_features + self.num_block_features * self.num_block_convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        """

        # TODO: Make more efficient by using keeping concatenated outputs

        # Store output of previous layers
        # Used as input of next layer
        outputs = [x]

        for block in self.blocks:
            # Forward propagation to single block
            x = block(x)

            # Store current block output
            outputs.append(x)

            # Concatenate all previous outputs as next input
            # Concatenate on channels
            x = torch.cat(outputs, dim=1)

        return x


class Dense(nn.Module):
    """
    GNINA Dense model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    num_blocks: int
        Numbr of dense blocks
    num_block_features: int
        Number of features in dense block convolutions
    num_block_convs" int
        Number of convolutions in dense block


    Notes
    -----
    Original implementation by Andrew McNutt available here:

        https://github.com/gnina/models/blob/master/pytorch/dense_model.py
    """

    def __init__(
        self,
        input_dims: Tuple,
        num_blocks: int = 3,
        num_block_features: int = 16,
        num_block_convs: int = 4,
    ) -> None:

        super().__init__()

        self.input_dims = input_dims

        features: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("data_enc_init_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                (
                    "data_enc_init_conv",
                    nn.Conv3d(
                        in_channels=input_dims[0],
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ),
                ("data_enc_init_conv_relu", nn.ReLU()),
            ]
        )

        out_features = 32
        for idx in range(num_blocks - 1):
            in_features = out_features

            # Dense block
            features[f"dense_block_{idx}"] = DenseBlock(
                in_features,
                num_block_features=num_block_features,
                num_block_convs=num_block_convs,
                tag=idx,
            )

            # Number of output features from dense block
            out_features = features[f"dense_block_{idx}"].out_features()

            features[f"data_enc_level{idx}_bottleneck"] = nn.Conv3d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=1,
                padding=0,
            )
            features[f"data_enc_level{idx}_bottleneck_relu"] = nn.ReLU()
            features[f"data_enc_level{idx+1}_pool"] = nn.MaxPool3d(
                kernel_size=2, stride=2
            )

        in_features = out_features
        features[f"dense_block_{num_blocks-1}"] = DenseBlock(
            in_features,
            num_block_features=num_block_features,
            num_block_convs=num_block_convs,
            tag=num_blocks - 1,
        )

        # Final number of channels
        self.features_out_size = features[f"dense_block_{num_blocks-1}"].out_features()

        # Final spatial dimensions (pre-global pooling)
        D = input_dims[1] // 2 ** num_blocks
        H = input_dims[2] // 2 ** num_blocks
        W = input_dims[3] // 2 ** num_blocks

        # Global MAX pooling
        # Redices spatial dimension to a single number per channel
        features[f"data_enc_level{num_blocks-1}_global_pool"] = nn.MaxPool3d(
            kernel_size=((D, H, W))
        )

        self.features = nn.Sequential(features)

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

    def forward(self, x):
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        affinity = self.affinity(x)

        return pose_raw, affinity
