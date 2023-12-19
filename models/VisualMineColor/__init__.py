# Load the model
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import base64
from PIL import Image
import datetime
from torchvision import transforms




IMAGES = 256

# Define the neural network architecture
class VisualMine(nn.Module):
    def __init__(self):
        super(VisualMine, self).__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.dropout1 = nn.Dropout(0.1)  
        self.fc2 = nn.Linear(1024, 2048)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(1024, 768)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


class Model:
    def __init__(self):
        self.model = VisualMine()
        path = os.path.dirname(__file__)
        self.model.load_state_dict(torch.load(f'{path}/VisualMineColor.pt'))
        self.model.eval()
    
    def add_noise(self,images, percent):
        if percent < 0 or percent > 1:
            raise ValueError("Percent must be between 0 and 1")
        noise = torch.randn(images.shape) * percent
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0, 1)
        return noisy_images

    def add_grayscale_contrast(self,images, contrast_percent):
        contrast_percent = torch.tensor(contrast_percent)
        images = images + (contrast_percent * (1 - images))
        images = torch.clamp(images, 0, 1)
        return images

    def generate(self, input, STEPS, NOISE, CONTRAST):
        transform = transforms.Compose([
        transforms.Resize((16, 16)),  # Resize to 16x16
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        image_tensor = transform(input)
        image_tensor = image_tensor.reshape(-1,768)
        basename = "image_"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        filename = "tmp/"+"_".join([basename, suffix]) + ".png" # e.g. 'mylogfile_120508_171442'

        with torch.no_grad():
            generated_images = self.model(image_tensor)
            for i in range(STEPS-1):
                generated_images = self.add_noise(generated_images, NOISE)
                generated_images = self.add_grayscale_contrast(generated_images, CONTRAST)

                generated_images = self.model(generated_images)

            base_images = generated_images.view(1, 3, 16, 16)
            save_image(base_images, filename)

        # Open the image and convert it to bytes
        with open(filename, 'rb') as image_file:
            image_bytes = image_file.read()

        os.remove(filename)
        # Convert the bytes to base64
        base64_bytes = base64.b64encode(image_bytes)

        # Convert the base64 bytes to string
        base64_string = base64_bytes.decode('utf-8')

        return base64_string

