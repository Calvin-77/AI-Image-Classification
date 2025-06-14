import torch
import timm
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path, num_classes):
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.head.in_features, num_classes)
    )
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model

def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5210, 0.4261, 0.3808), std=(0.2769, 0.2514, 0.2524)),
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def infer(image, model, class_names):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze()
        predicted_idx = torch.argmax(probabilities).item()
    return predicted_idx, probabilities.cpu().tolist()

def predict_ui(image):
    checkpoint_path = "./output_models/vit_small-ckpt.t7"
    class_names = ['Fake', 'Real']
    model = load_model(checkpoint_path, num_classes=len(class_names))
    predicted_idx, probabilities = infer(image, model, class_names)
    result = f"Prediction: {class_names[predicted_idx]}\n"
    result += "Class Probabilities:\n"
    for idx, prob in enumerate(probabilities):
        result += f"{class_names[idx]}: {prob * 100:.2f}%\n"
    return result

interface = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="text",
    title="AI vs Human Image Detection",
    description="Upload an image to detect whether it is AI-generated or human-made."
)

if __name__ == "__main__":
    interface.launch()
