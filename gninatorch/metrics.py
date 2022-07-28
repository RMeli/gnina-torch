from typing import Any, Dict

import torch
from ignite import metrics
from ignite.contrib.metrics import ROC_AUC
from torch import nn

from gninatorch import transforms


def setup_metrics(
    affinity: bool,
    flex: bool,
    pose_loss: nn.Module,
    affinity_loss: nn.Module,
    flexpose_loss: nn.Module,
    roc_auc: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Define metrics to be computed at the end of an epoch (evaluation).

    Parameters
    ----------
    affinity: bool
        Flag for affinity prediction (in addition to ligand pose prediction)
    flex: bool
        Flag for flexible residues pose prediction (in addition to ligand pose
        prediction)
    pose_loss: nn.Module
        Pose loss
    affinity_loss: nn.Module
        Affinity loss
    flexpose_loss: nn.Module
        Flexible residues pose loss
    roc_auc: bool
        Flag for computing ROC AUC
    device: torch.device
        Device

    Returns
    -------
    Dict[str, ignite.metrics.Metric]
        Dictionary of PyTorch Ignite metrics

    Notes
    -----
    The computation of the ROC AUC for pose prediction can be disabled. This is useful
    when the computation is expected to fail because all poses belong to the same class
    (e.g. all poses are "good" poses). This situations happens when working with crystal
    structures, for which the pose is a "good" pose by definition.

    Loss functions need to be set up as metrics in order to be correctly accumulated.
    Using :code:`evaluator.state.output` to compute the loss does not work since the
    output only contain the last batch (to avoid RAM saturation).
    """
    # Check that affinity_loss and affinity arguments are consistent
    if affinity_loss is not None:
        assert affinity

    # Check that flexpose_loss and flex arguments are consistent
    if flexpose_loss is not None:
        assert flex

    # Check that either affinity or flex is set
    assert not (affinity and flex)

    # Pose prediction metrics
    m: Dict[str, Any] = {
        # Accuracy can be used directly without binarising the data since we are not
        # performing binary classification (Linear(out_features=1)) but we are
        # performing multiclass classification with 2 classes (Linear(out_features=2))
        "Accuracy": metrics.Accuracy(
            output_transform=transforms.output_transform_select_pose
        ),
        # Balanced accuracy is the average recall over all classes
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
        "Balanced Accuracy": metrics.Recall(
            average=True, output_transform=transforms.output_transform_select_pose
        ),
    }

    if pose_loss is not None:
        # For the loss function, log_softmax is needed as opposed to softmax
        # Use transforms.output_transform_select_log_pose instead of
        # transforms.output_transform_select_pose
        m.update(
            {
                "Pose Loss": metrics.Loss(
                    pose_loss,
                    output_transform=transforms.output_transform_select_log_pose,
                )
            }
        )

    if roc_auc:
        m.update(
            {
                "ROC AUC": ROC_AUC(
                    output_transform=lambda output: transforms.output_transform_ROC(
                        output
                    ),
                    device=device,
                ),
            }
        )

    # Affinity prediction metrics
    if affinity:
        # Affinities have negative values for bad poses
        # In order to compute metrics, the absolute value is returned
        m.update(
            {
                "MAE": metrics.MeanAbsoluteError(
                    output_transform=transforms.output_transform_select_affinity_abs
                ),
                "RMSE": metrics.RootMeanSquaredError(
                    output_transform=transforms.output_transform_select_affinity_abs
                ),
            }
        )

        if affinity_loss is not None:
            # Affinities have negative values for bad poses
            # The loss function uses the sign to distinguish good from bad poses
            m.update(
                {
                    "Affinity Loss": metrics.Loss(
                        affinity_loss,
                        output_transform=transforms.output_transform_select_affinity,
                    )
                }
            )

    # Flexible residues pose prediction metrics
    if flex:
        # Pose prediction metrics
        m.update(
            {
                # Accuracy can be used directly without binarising the data since we are not
                # performing binary classification (Linear(out_features=1)) but we are
                # performing multiclass classification with 2 classes (Linear(out_features=2))
                "Flex Accuracy": metrics.Accuracy(
                    output_transform=transforms.output_transform_select_flex
                ),
                # Balanced accuracy is the average recall over all classes
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
                "Flex Balanced Accuracy": metrics.Recall(
                    average=True,
                    output_transform=transforms.output_transform_select_flex,
                ),
            }
        )

        if roc_auc:
            m.update(
                {
                    "Flex ROC AUC": ROC_AUC(
                        output_transform=lambda output: transforms.output_transform_ROC_flex(
                            output
                        ),
                        device=device,
                    ),
                }
            )

        if flexpose_loss is not None:
            # For the loss function, log_softmax is needed as opposed to softmax
            # Use transforms.output_transform_select_log_flex instead of
            # transforms.output_transform_select_flex
            m.update(
                {
                    "Flex Pose Loss": metrics.Loss(
                        flexpose_loss,
                        output_transform=transforms.output_transform_select_log_flex,
                    )
                }
            )

    # Return dictionary with all metrics
    return m
