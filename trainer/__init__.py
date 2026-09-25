import torch
from enum import Enum

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import Accuracy, F1Score, AUROC, MatthewsCorrCoef

class Tasks(Enum):
    SEG = "seg"
    REC = "rec"
    CLA = "cla"


def get_metrics(task, test_stage=False, num_seg_classes=3, multilabel=True, classification_task="binary", num_classes=2):
    if task == Tasks.CLA.value:
        average = "none" if classification_task == "binary" else "macro"
        metrics = torch.nn.ModuleDict(
            {
                "acc": Accuracy(
                    task=classification_task,
                    num_classes=num_classes,
                    average=average,
                ),
                "f1": F1Score(
                    task=classification_task,
                    num_classes=num_classes,
                    average=average,
                ),
                "auroc": AUROC(
                    task=classification_task,
                    num_classes=num_classes,
                    average=average,
                ),
                "mcc": MatthewsCorrCoef(
                    task=classification_task,
                    num_classes=num_classes,
                ),
            }
        )
    elif task == Tasks.SEG.value:
        average = "none" if test_stage else "macro"
        metrics = torch.nn.ModuleDict(
            {
                "dice": DiceScore(
                    num_classes=num_seg_classes,
                    average=average,
                    input_format="one-hot" if multilabel else "index",
                ),
                "mIoU": MeanIoU(
                    num_classes=num_seg_classes,
                    input_format="one-hot" if multilabel else "index",
                ),
            }
        )
    elif task == Tasks.REC.value:
        metrics = torch.nn.ModuleDict(
            {
                "ssim": StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)),
                "psnr": PeakSignalNoiseRatio(data_range=(-1.0, 1.0)),
            }
        )
    return metrics

