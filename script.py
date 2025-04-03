import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

class ICLightV2(torch.nn.Module):
    def __init__(self):
        super(ICLightV2, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.layers(x)

def load_model():
    model = ICLightV2()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return output_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Free Unlimited AI Image Enhancer')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save enhanced image')
    args = parser.parse_args()
    
    model = load_model()
    output_image = process_image(args.input, model)
    output_image.save(args.output)
    print(f"Enhanced image saved at {args.output}")
