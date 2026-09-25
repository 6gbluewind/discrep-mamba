from trainer import Tasks
from loss.loss import ClassLoss, SegLoss, PixelLoss
from loss.md_loss import ModalityDifferenceLoss

def get_loss(task, device, num_classes, in_channels, diffloss=False, *args, **kwargs):
    if task == Tasks.CLA.value:
        print("use cls loss")
        return ClassLoss(device=device,num_classes=num_classes)
    elif task == Tasks.SEG.value:
        print("seg loss")
        return SegLoss(device=device)
    elif task == Tasks.REC.value and diffloss:
        print("use SAFC loss")
        return ModalityDifferenceLoss(in_channels, device=device)
    elif task == Tasks.REC.value and not diffloss:
        print("use pixel loss")
        return PixelLoss(device=device)
