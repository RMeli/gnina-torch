"""
GNINA Caffe models translated to PyTorch.

Notes
-----
The PyTorch models try to follow the original Caffe models as much as possible. However,
some changes are necessary.

Notable differences:
* The :code:`MolDataLayer` is now separated from the model and the parameters are
controlled by CLI arguments in the training process.
* The model output for pose prediction corresponds to the log softmax of the last fully-
connected layer instead of the softmax.
"""

from collections import OrderedDict, namedtuple
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


def weights_and_biases_init(m: nn.Module) -> None:
    """
    Initialize the weights and biases of the model.

    Parameters
    ----------
    m : nn.Module
        Module (layer) to initialize

    Notes
    -----
    This function is used to initialize the weights of the model for both convolutional
    and linear layers. Weights are initialized using uniform Xavier initialization
    while biases are set to zero.

    https://github.com/gnina/libmolgrid/blob/e6d5f36f1ae03f643ca69cdec1625ac52e653f88/test/test_torch_cnn.py#L45-L48
    """
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


class Default2017(nn.Module):
    """
    GNINA default2017 model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2017.model

    The main difference is that the PyTorch implementation resurns the log softmax.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        assert (
            len(input_dims) == 4
        ), "Input dimensions must be (channels, depth, height, width)"

        self.input_dims = input_dims

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # unit1
                    ("unit1_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit1_conv1",
                        nn.Conv3d(
                            in_channels=input_dims[0],
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit1_relu1", nn.ReLU()),
                    # unit2
                    ("unit2_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit2_conv1",
                        nn.Conv3d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit2_relu1", nn.ReLU()),
                    # unit3
                    ("unit3_pool", nn.MaxPool3d(kernel_size=2, stride=2)),
                    (
                        "unit3_conv1",
                        nn.Conv3d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit3_relu1", nn.ReLU()),
                ]
            )
        )

        self.features_out_size = (
            input_dims[1] // 8 * input_dims[2] // 8 * input_dims[3] // 8 * 128
        )

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class Default2017Pose(Default2017):
    """
    GNINA default2017 model architecture for pose prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2017.model

    The main difference is that the PyTorch implementation resurns the log softmax of
    the final linear layer instead of feeding it to a :code:`SoftmaxWithLoss` layer.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for pose prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Log probabilities for ligand pose

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        return pose_log


class Default2017Affinity(Default2017Pose):
    """
    GNINA default2017 model architecture for pose and affinity prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2017.model

    The main difference is that the PyTorch implementation resurns the log softmax of
    the final linear layer instead of feeding it to a :code:`SoftmaxWithLoss` layer.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for binding affinity prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and affinity prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        affinity = self.affinity(x)
        # Squeeze last (dummy) dimension of affinity prediction
        # This allows to match the shape (batch_size,) of the target tensor
        return pose_log, affinity.squeeze(-1)


class Default2017Flex(Default2017):
    """
    GNINA default2017 model architecture for multi-task pose prediction (ligand and
    flexible residues).

    Poses are annotated based on both ligand RMSD and flexible residues RMSD (w.r.t. the
    cognate receptor in the case of cross-docking).

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for ligand pose prediction
        self.lig_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lig_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

        # Linear layer for flexible residues pose prediction
        self.flex_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "flex_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and flexible residues pose prediction

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and flexible residues pose prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        lig_pose_raw = self.lig_pose(x)
        lig_pose_log = F.log_softmax(lig_pose_raw, dim=1)

        flex_pose_raw = self.flex_pose(x)
        flex_pose_log = F.log_softmax(flex_pose_raw, dim=1)

        return lig_pose_log, flex_pose_log


class Default2018(nn.Module):
    """
    GNINA default2017 model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2018.model

    The main difference is that the PyTorch implementation resurns the log softmax.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        assert (
            len(input_dims) == 4
        ), "Input dimensions must be (channels, depth, height, width)"
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        """
        raise NotImplementedError


class Default2018Pose(Default2018):
    """
    GNINA default2017 model architecture for pose prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2018.model

    The main difference is that the PyTorch implementation resurns the log softmax.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for pose prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Log probabilities for ligand pose

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        return pose_log


class Default2018Affinity(Default2018Pose):
    """
    GNINA default2017 model architecture for pose and affinity prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/default2018.model

    The main difference is that the PyTorch implementation resurns the log softmax.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for binding affinity prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and affinity prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        affinity = self.affinity(x)
        # Squeeze last (dummy) dimension of affinity prediction
        # This allows to match the shape (batch_size,) of the target tensor
        return pose_log, affinity.squeeze(-1)


class Default2018Flex(Default2018):
    """
    GNINA default2017 model architecture for multi-task pose prediction (ligand and
    flexible residues).

    Poses are annotated based on both ligand RMSD and flexible residues RMSD (w.r.t. the
    cognate receptor in the case of cross-docking).

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    """

    def __init__(self, input_dims: Tuple):

        super().__init__(input_dims)

        # Linear layer for ligand pose prediction
        self.lig_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lig_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

        # Linear layer for flexible residues pose prediction
        self.flex_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "flex_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and flexible residues pose prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """

        x = self.features(x)
        x = x.view(-1, self.features_out_size)

        lig_pose_raw = self.lig_pose(x)
        lig_pose_log = F.log_softmax(lig_pose_raw, dim=1)

        flex_pose_raw = self.flex_pose(x)
        flex_pose_log = F.log_softmax(flex_pose_raw, dim=1)

        return lig_pose_log, flex_pose_log


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
    The total number of output features corresponds to the input features concatenated
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

        dense_dict: OrderedDict[str, nn.Module] = OrderedDict()

        self.in_features = in_features
        self.num_block_features = num_block_features
        self.num_block_convs = num_block_convs

        in_features_layer = in_features
        for idx in range(num_block_convs):
            dense_dict.update(
                [
                    (
                        f"data_enc_level{tag}_batchnorm_conv{idx}",
                        nn.BatchNorm3d(
                            in_features_layer,
                            affine=True,  # Same effect as "Scale" layer in Caffe
                        ),
                    ),
                    (
                        f"data_enc_level{tag}_conv{idx}",
                        nn.Conv3d(
                            in_channels=in_features_layer,
                            out_channels=num_block_features,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (f"data_enc_level{tag}_conv{idx}_relu", nn.ReLU()),
                ]
            )

            # The next layer takes all previous features as input
            in_features_layer += num_block_features

        self.blocks = nn.Sequential(dense_dict)

    def out_features(self) -> int:
        return self.in_features + self.num_block_features * self.num_block_convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        # TODO: Make more efficient by keeping concatenated outputs

        # Store output of previous layers
        # Used as input of next layer
        outputs = [x]

        for block in self.blocks:
            # Forward propagation to single block
            x = block(x)

            if isinstance(block, nn.ReLU):
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
        Number of dense blocks
    num_block_features: int
        Number of features in dense block convolutions
    num_block_convs" int
        Number of convolutions in dense block


    Notes
    -----
    Original implementation by Andrew McNutt available here:

        https://github.com/gnina/models/blob/master/pytorch/dense_model.py

    The main difference is that the original implementation returns the raw output of
    the last linear layer while here the output is the log softmax of the last linear.
    """

    def __init__(
        self,
        input_dims: Tuple,
        num_blocks: int = 3,
        num_block_features: int = 16,
        num_block_convs: int = 4,
        affinity: bool = True,
    ) -> None:

        super().__init__()

        assert (
            len(input_dims) == 4
        ), "Input dimensions must be (channels, depth, height, width)"
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

        out_features: int = 32
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
        D = input_dims[1] // 2**num_blocks
        H = input_dims[2] // 2**num_blocks
        W = input_dims[3] // 2**num_blocks

        # Global MAX pooling
        # Redices spatial dimension to a single number per channel
        features[f"data_enc_level{num_blocks-1}_global_pool"] = nn.MaxPool3d(
            kernel_size=((D, H, W))
        )

        self.features = nn.Sequential(features)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Raises
        ------
        NotImplementedError

        Notes
        -----
        The forward pass needs to be implemented in derived classes.
        """
        raise NotImplementedError


class DensePose(Dense):
    """
    GNINA Dense model architecture for pose prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    num_blocks: int
        Number of dense blocks
    num_block_features: int
        Number of features in dense block convolutions
    num_block_convs" int
        Number of convolutions in dense block


    Notes
    -----
    Original implementation by Andrew McNutt available here:

        https://github.com/gnina/models/blob/master/pytorch/dense_model.py

    The main difference is that the original implementation resurns the raw output of
    the last linear layer while here the output is the log softmax of the last linear.
    """

    def __init__(
        self,
        input_dims: Tuple,
        num_blocks: int = 3,
        num_block_features: int = 16,
        num_block_convs: int = 4,
    ) -> None:

        super().__init__(input_dims, num_blocks, num_block_features, num_block_convs)

        # Linear layer for binding pose prediction
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

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Log probabilities for ligand pose

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        return pose_log


class DenseAffinity(DensePose):
    """
    GNINA Dense model architecture for binding affinity prediction.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    num_blocks: int
        Number of dense blocks
    num_block_features: int
        Number of features in dense block convolutions
    num_block_convs" int
        Number of convolutions in dense block


    Notes
    -----
    Original implementation by Andrew McNutt available here:

        https://github.com/gnina/models/blob/master/pytorch/dense_model.py

    The main difference is that the original implementation resurns the raw output of
    the last linear layer while here the output is the log softmax of the last linear.
    """

    def __init__(
        self,
        input_dims: Tuple,
        num_blocks: int = 3,
        num_block_features: int = 16,
        num_block_convs: int = 4,
    ) -> None:

        super().__init__(input_dims, num_blocks, num_block_features, num_block_convs)

        # Linear layer for binding affinity prediction
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

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and affinity prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        affinity = self.affinity(x)
        # Squeeze last (dummy) dimension of affinity prediction
        # This allows to match the shape (batch_size,) of the target tensor
        return pose_log, affinity.squeeze(-1)


class DenseFlex(Dense):
    """
    GNINA dense model architecture for multi-task pose prediction (ligand and
    flexible residues).

    Poses are annotated based on both ligand RMSD and flexible residues RMSD (w.r.t. the
    cognate receptor in the case of cross-docking).

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)
    num_blocks: int
        Number of dense blocks
    num_block_features: int
        Number of features in dense block convolutions
    num_block_convs" int
        Number of convolutions in dense block


    Notes
    -----
    Original implementation by Andrew McNutt available here:

        https://github.com/gnina/models/blob/master/pytorch/dense_model.py

    The main difference is that the original implementation resurns the raw output of
    the last linear layer while here the output is the log softmax of the last linear.
    """

    def __init__(
        self,
        input_dims: Tuple,
        num_blocks: int = 3,
        num_block_features: int = 16,
        num_block_convs: int = 4,
    ) -> None:

        super().__init__(input_dims, num_blocks, num_block_features, num_block_convs)

        # Linear layer for ligand pose prediction
        self.lig_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lig_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

        # Linear layer for flexible residues pose prediction
        self.flex_pose = nn.Sequential(
            OrderedDict(
                [
                    (
                        "flex_pose_output",
                        nn.Linear(in_features=self.features_out_size, out_features=2),
                    )
                ]
            )
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and flexible residues pose prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        lig_pose_raw = self.lig_pose(x)
        lig_pose_log = F.log_softmax(lig_pose_raw, dim=1)

        flex_pose_raw = self.flex_pose(x)
        flex_pose_log = F.log_softmax(flex_pose_raw, dim=1)

        return lig_pose_log, flex_pose_log


class HiResPose(nn.Module):
    """
    GNINA HiResPose model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/hires_pose.model

    The main difference is that the PyTorch implementation resurns the log softmax.

    This model is implemented only for multi-task pose and affinity prediction.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        self.input_dims = input_dims

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # unit1
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

        # Two MaxPool3d layers with kernel_size=2 and stride=2
        # Spatial dimensions are halved at each pooling step
        self.features_out_size = (
            input_dims[1] // 4 * input_dims[2] // 4 * input_dims[3] // 4 * 128
        )

        # Linear layer for pose prediction
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

        # Linear layer for binding affinity prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and affinity prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        affinity = self.affinity(x)
        # Squeeze last (dummy) dimension of affinity prediction
        # This allows to match the shape (batch_size,) of the target tensor
        return pose_log, affinity.squeeze(-1)


class HiResAffinity(nn.Module):
    """
    GNINA HiResAffinity model architecture.

    Parameters
    ----------
    input_dims: tuple
        Model input dimensions (channels, depth, height, width)

    Notes
    -----
    This architecture was translated from the following Caffe model:

        https://github.com/gnina/models/blob/master/crossdocked_paper/hires_pose.model

    The main difference is that the PyTorch implementation resurns the log softmax.

    This model is implemented only for multi-task pose and affinity prediction.
    """

    def __init__(self, input_dims: Tuple):

        super().__init__()

        self.input_dims = input_dims

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # unit1
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
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("unit2_func", nn.ReLU()),
                    # unit3
                    ("unit3_pool", nn.AvgPool3d(kernel_size=8, stride=8)),
                    (
                        "unit3_conv",
                        nn.Conv3d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                        ),
                    ),
                    ("unit3_func", nn.ELU(alpha=1.0)),
                    # unit5 (following original naming convention)
                    ("unit5_pool", nn.MaxPool3d(kernel_size=4, stride=4)),
                ]
            )
        )

        self.features_out_size = (
            input_dims[1]
            // (8 * 4)
            * input_dims[2]
            // (8 * 4)
            * input_dims[3]
            // (8 * 4)
            * 128
        )

        # Linear layer for pose prediction
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

        # Linear layer for binding affinity prediction
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

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Log probabilities for ligand pose and affinity prediction

        Notes
        -----
        The pose score is the log softmax of the output of the last linear layer.
        """
        x = self.features(x)

        # Reshape based on number of channels
        # Global max pooling reduced spatial dimensions to single value
        x = x.view(-1, self.features_out_size)

        pose_raw = self.pose(x)
        pose_log = F.log_softmax(pose_raw, dim=1)

        affinity = self.affinity(x)
        # Squeeze last (dummy) dimension of affinity prediction
        # This allows to match the shape (batch_size,) of the target tensor
        return pose_log, affinity.squeeze(-1)


Model = namedtuple("Model", ["model", "affinity", "flex"])

# Key: model name, affinity, flexible residues
models_dict = {
    Model("default2017", False, False): Default2017Pose,
    Model("default2017", True, False): Default2017Affinity,
    Model("default2017", False, True): Default2017Flex,
    Model("default2018", False, False): Default2018Pose,
    Model("default2018", True, False): Default2018Affinity,
    Model("default2018", False, True): Default2018Flex,
    Model("dense", False, False): DensePose,
    Model("dense", True, False): DenseAffinity,
    Model("dense", False, True): DenseFlex,
    Model("hires_pose", True, False): HiResPose,
    Model("hires_affinity", True, False): HiResAffinity,
}


class GNINAModelEnsemble(nn.Module):
    """
    Ensemble of GNINA models.

    Parameters
    ----------
    models: List[nn.Module]
        List of models to use in the ensemble

    Notes
    -----
    Assume models perform only pose AND affinity prediction.

    Modules are stored in :code:`nn.ModuleList` so that they are properly registered.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()

        # Check that all models allow both pose and affinity predictions
        # These are the only models supported by GNINA so far
        for m in models:
            assert (
                isinstance(m, Default2017Affinity)
                or isinstance(m, Default2018Affinity)
                or isinstance(m, DenseAffinity)
            )

        # nn.ModuleList allows to register the different modules
        # This makes things like .to(device) apply to all modules
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, torch.tensor],
            Logarithm of the pose score, affinity prediction (average) and affinity
            variance

        Notes
        -----
        For pose prediction, the average has to be performed on the scores, not theeir
        logarithm (returned by the model). In order to be consistent with everywhere
        else (where the logarighm of the prediction is returned), here we compute the
        score (by exponentating), compute the average, and finally return the logarithm
        of the computed average.
        """
        predictions = [model(x) for model in self.models]

        # map(list, zip(*predictions)) transform list of multi-task predictions into
        # list of predictions for each task
        # [(log_pose_1, affinity_1), (log_pose_2, affinity_2), ...] =>
        # [[log_pose_1, log_pose_2, ...], [affinity_1, affinity_2, ...]]
        # Suggested by @IAlibay
        # TODO: Better way to do this?
        log_pose_all, affinity_all = tuple(map(list, zip(*predictions)))

        affinity_stacked = torch.stack(affinity_all)

        log_pose_avg = torch.stack(log_pose_all).exp().mean(dim=0).log()
        affinity_avg = affinity_stacked.mean(dim=0)
        affinity_var = affinity_stacked.var(dim=0, unbiased=False)

        return log_pose_avg, affinity_avg, affinity_var
