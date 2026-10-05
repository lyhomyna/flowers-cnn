import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from model import FlowerCNN, FlowerCNNDropout, FlowerCNNBatchNorm

output_folder = "output_folder"
plots_folder = "plots"

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Варіант моделі для порівняння (п.8-9 завдання): "base" | "dropout" | "batchnorm"
VARIANT = "base"
DROPOUT_P = 0.5

if VARIANT == "dropout":
    PLOT_SUFFIX = f"_dropout_{DROPOUT_P}"
elif VARIANT == "batchnorm":
    PLOT_SUFFIX = "_batchnorm"
else:
    PLOT_SUFFIX = ""

transform = transforms.ToTensor()

train_dataset = datasets.ImageFolder(os.path.join(output_folder, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(output_folder, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(output_folder, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)

if VARIANT == "dropout":
    model = FlowerCNNDropout(num_classes=num_classes, dropout_p=DROPOUT_P).to(DEVICE)
elif VARIANT == "batchnorm":
    model = FlowerCNNBatchNorm(num_classes=num_classes).to(DEVICE)
else:
    model = FlowerCNN(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total

def predict_with_probs(model, loader, device):
    model.eval()
    y_true, y_pred, probs, imgs = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            p = torch.softmax(outputs, dim=1)
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
            probs.extend(p.cpu().tolist())
            imgs.extend(images.cpu())
    return y_true, y_pred, probs, imgs

if __name__ == "__main__":
    print(f"Варіант моделі: {VARIANT}" + (f" (dropout_p={DROPOUT_P})" if VARIANT == "dropout" else ""))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    os.makedirs(plots_folder, exist_ok=True)

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy per epoch ({VARIANT})")
    plt.legend()
    plt.savefig(os.path.join(plots_folder, f"accuracy{PLOT_SUFFIX}.png"))

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss per epoch ({VARIANT})")
    plt.legend()
    plt.savefig(os.path.join(plots_folder, f"loss{PLOT_SUFFIX}.png"))

    # Оцінка на test (п.6 завдання): accuracy, precision, recall, F1-score, confusion matrix
    y_true, y_pred, probs, imgs = predict_with_probs(model, test_loader, DEVICE)
    class_names = test_dataset.classes

    test_acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nTest: accuracy={test_acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion matrix ({VARIANT}, test)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"confusion_matrix{PLOT_SUFFIX}.png"))

    # Приклади помилок з ймовірностями (п.7 завдання)
    misclassified = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    print(f"Помилок на test: {len(misclassified)} з {len(y_true)}")

    sample_idx = misclassified[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, idx in zip(axes.flat, sample_idx):
        img = imgs[idx].permute(1, 2, 0).numpy()
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        pred_prob = probs[idx][y_pred[idx]]
        ax.imshow(img)
        ax.set_title(f"true: {true_name}\npred: {pred_name} (p={pred_prob:.2f})", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(sample_idx):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"misclassified_examples{PLOT_SUFFIX}.png"))

    # Збереження моделі (п.10 завдання) — зручно мати для кожного варіанту
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", f"model{PLOT_SUFFIX or '_base'}.pt"))
