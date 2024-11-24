import torch
from segmentation_models_pytorch.metrics import iou_score,get_stats,f1_score

def EVAL_SCORES(outputs, masks, num_classes):
    tp, fp, fn, tn = get_stats(
        outputs.softmax(dim=1).argmax(dim=1), torch.argmax(masks, dim=1),
        mode="multiclass", num_classes=num_classes
    )
    per_batch_iou = iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    per_batch_f1 =  f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    return per_batch_iou, per_batch_f1