import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import torch


def plot_axis(train_loss=None, test_loss=None, train_accur = None, test_accur=None):
    fig, axs = plt.subplots(1,2,figsize=(18,8))
    axs[0].set_title('LOSS')
    axs[0].plot(train_loss, label='Train')
    axs[0].plot(test_loss, label='Test')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].plot(train_accur, label='Train')
    axs[1].plot(test_accur, label='Test')
    axs[1].legend()
    axs[1].grid()

    plt.show()

def plot_Lossaxis(train_loss=None, test_loss=None):
    fig, axs = plt.subplots(1,2,figsize=(18,8))
    axs[0].set_title('Train-Loss')
    axs[0].plot(train_loss, label='Train')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Test-Loss')
    axs[1].plot(test_loss, label='Test')
    axs[1].legend()
    axs[1].grid()

    plt.show()
 
def plot_Accaxis(train_acc=None, test_Acc=None):
  fig, axs = plt.subplots()
  axs.set_title('Accuracy')
  axs.plot(train_acc, label='Train')
  axs.plot(test_Acc, label='Test')
  axs.legend()
  axs.grid()

def misclassified_images(model, test_loader, device, class_names=None, n_images=20):
    """
    Get misclassified images.
    """
    wrong_images = []
    wrong_label = []
    correct_label = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = pred.eq(target.view_as(pred)) == False
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f"Total wrong predictions are {len(wrong_predictions)}")

        plot_misclassified_images(wrong_predictions, class_names=class_names, n_images=n_images)

    return wrong_predictions


def plot_misclassified_images(wrong_predictions, n_images=20, class_names=None):
    """
    Plot the misclassified images.
    """
    if class_names is None:
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig = plt.figure(figsize=(10, 12))
    fig.tight_layout()
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    for i, (img, pred, correct) in enumerate(wrong_predictions[:n_images]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j] * std[j]) + mean[j]

        img = np.transpose(img, (1, 2, 0))
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        ax.set_title(f"\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}", fontsize=10)
        ax.imshow(img)

    plt.show()