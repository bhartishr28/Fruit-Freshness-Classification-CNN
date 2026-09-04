from PIL import Image
import torch
import torch.nn as nn

from torchvision import models, transforms

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = None
num_classes=16
# display_names = ['Fresh Banana','Fresh Lemon','Fresh Lulo','Fresh Mango',
#  'Fresh Orange','Fresh Strawberry','Fresh Tamarillo','Fresh Tomato',
#  'Spoiled Banana','Spoiled Lemon','Spoiled Lulo','Spoiled Mango',
#  'Spoiled Orange','Spoiled Strawberry','Spoiled Tamarillo','Spoiled Tomato']

class_names=[ "F_Banana",
    "F_Lemon",
    "F_Lulo",
    "F_Mango",
    "F_Orange",
    "F_Strawberry",
    "F_Tamarillo",
    "F_Tomato",
    "S_Banana",
    "S_Lemon",
    "S_Lulo",
    "S_Mango",
    "S_Orange",
    "S_Strawberry",
    "S_Tamarillo",
    "S_Tomato"]

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_trained_model(checkpoint_path="best_resnet50_freshharvest.pth"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(device)

    model.eval()
    return model


# Prediction function
def predict_fruit(model,image):
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        predicted_class = torch.argmax(outputs, dim=1)

    predicted_label = class_names[predicted_class.item()]

    return predicted_label
