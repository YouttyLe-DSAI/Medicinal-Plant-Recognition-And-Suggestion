import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse  # Import argparse for command-line arguments

from plant_recognition_project.src.model import CustomCNN  # Import CustomCNN from model.py

def load_data(data_dir, batch_size, transform, val_split=0.2):
    """Loads and preprocesses image datasets, splitting into training and validation sets."""
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(dataset.classes)
    train_size = int((1.0 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, num_classes, dataset.classes

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, early_stopping_patience):
    """Trains the model, performs validation, and saves the best model."""
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)  # Save best model
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    """Plots training and validation loss and accuracy curves."""
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)

    plt.plot(epochs, train_accuracies, 'b-', marker='o', label="Train Accuracy")
    plt.plot(epochs, val_accuracies, 'g-', marker='s', label="Validation Accuracy")
    plt.plot(epochs, train_losses, 'r-', marker='o', label="Train Loss")
    plt.plot(epochs, val_losses, 'orange', marker='s', label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training & Validation Metrics")
    plt.legend()
    plt.grid()
    plt.show()

def main(args):
    """Main function to train the plant recognition model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([ # Define transformations
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader, val_loader, num_classes, class_names = load_data( # Load data
        args.data_dir, args.batch_size, transform, args.val_split)
    print(f"Number of classes: {num_classes}, Class names: {class_names}")

    model = CustomCNN(num_classes).to(device) # Initialize model
    summary(model, (3, 224, 224)) # Print model summary

    criterion = nn.CrossEntropyLoss() # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=0.1)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model( # Train model
        model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device, args.model_save_path, args.early_stopping_patience)

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, args.epochs) # Plot metrics
    print(f"Best model saved to: {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Recognition Model Training Script")
    parser.add_argument('--data_dir', type=str, default='D:\FPT_SUBJECT\Term 4\DAP391m\DAP_Project\plant_recognition_project\data\sorted_data_aug', help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_patience', type=int, default=2, help='Patience for learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--model_save_path', type=str, default='models/best_model.pth', help='Path to save the best model')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')

    args = parser.parse_args()
    main(args)