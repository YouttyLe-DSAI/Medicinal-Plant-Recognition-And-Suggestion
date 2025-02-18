import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse # Import argparse
from plant_recognition_project.src.model import CustomCNN # Import CustomCNN from model.py

def predict_image(image_path, model, class_names, device, transform, confidence_threshold=0.8):
    """Predicts the class of an image using the trained model."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_index = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class_index].item()

            if confidence >= confidence_threshold:
                predicted_label = class_names[predicted_class_index]
                return f"Đây là vị thuốc: {predicted_label} (Độ tin cậy: {confidence:.2f})"
            else:
                return "Không tìm thấy loại thuốc phù hợp (Độ tin cậy thấp)."

    except FileNotFoundError:
        return "Không tìm thấy ảnh."
    except Exception as e:
        return f"Lỗi: {e}"

def main(args):
    """Main function to load model, class names, and predict an image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir # Đường dẫn dataset để lấy tên lớp
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)

    model = CustomCNN(num_classes).to(device) # Load model
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    transform = transforms.Compose([ # Define transformations (giữ nguyên transform train)
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    prediction_result = predict_image(args.image_path, model, class_names, device, transform, args.confidence_threshold)
    print(prediction_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Recognition Image Prediction Script")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file for prediction')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, default='data/sorted_data_aug', help='Path to the dataset directory (for class names)')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Confidence threshold for prediction')

    args = parser.parse_args()
    main(args)