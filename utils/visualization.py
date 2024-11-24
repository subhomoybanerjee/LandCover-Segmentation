import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def visualizePreds(model, test_loader,device,predictions_dir):
    images, masks = next(iter(test_loader))
    images, masks = images.to(device), masks.to(device)
    with torch.inference_mode():
        model.eval()
        logits = model(images)

    pr_masks = logits.softmax(dim=1)
    pr_masks = pr_masks.argmax(dim=1)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx <= 20:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(np.argmax(np.moveaxis(gt_mask.cpu().numpy(), 0, 2), axis=-1),
                       cmap="viridis")
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(pr_mask.cpu().numpy(), cmap="viridis")
            plt.title("Prediction")
            plt.axis("off")

            plt.savefig(os.path.join(predictions_dir, f'prediction_{idx}.png'), bbox_inches='tight')
            plt.close()
        else:
            break
    print(f'predicted images saved in {predictions_dir}\n')
def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure()
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} Over Epochs')
    plt.savefig(os.path.join(save_path,f'{metric_name}.png'))
    plt.close()