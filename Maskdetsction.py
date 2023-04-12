import torch

import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(

    nn.Conv2d(3, 16, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(64*18*18, 256),

    nn.ReLU(),

    nn.Linear(256, 2),

    nn.Softmax(dim=1)

)

model.load_state_dict(torch.load('maskvsnomask.pth', map_location=device))

model.to(device)

model.eval()

img_path = 'Masks Dataset/Validation/Non Mask/real_00001.jpg'

img = Image.open(img_path)

transform = transforms.Compose([

    transforms.Resize((150, 150)),

    transforms.ToTensor(),

])

img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():

    prediction = model(img_tensor)

    print(prediction)

