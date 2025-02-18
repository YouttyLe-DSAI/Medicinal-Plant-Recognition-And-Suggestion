import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os



class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Đường dẫn dữ liệu và tên lớp ---
data_dir = r"C:\Users\ACER\Desktop\FPTU\Spring25\DAP391m\Best_CNN\sorted_data_aug"
class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes).to(device)

# *** SỬA CHỮA: Sử dụng đúng đường dẫn và không dùng weights_only ***
model.load_state_dict(torch.load("best_model.pth", map_location=device), strict=False)


model.eval()

# --- Transform Ảnh ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load và Dự Đoán Ảnh ---
img_path = r"C:\Users\ACER\Desktop\FPTU\Spring25\DAP391m\Best_CNN\Data_Test\kimvang.jpg"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# --- Dự đoán ---
with torch.no_grad():
    output = model(image)
    predicted_class_index = torch.argmax(output, dim=1).item()
    predicted_label = class_names[predicted_class_index]

print(f"Ảnh được dự đoán thuộc lớp: {predicted_label} (index: {predicted_class_index})")