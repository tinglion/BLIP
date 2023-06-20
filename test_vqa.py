from models.blip_vqa import blip_vqa
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(image_size, device):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    raw_image = Image.open('D:\\projects\\3\\ai\\VisualGLM-6B\\jiao_data\\V1\\test.jpg')

    w, h = raw_image.size
    # display(raw_image.resize((w//5, h//5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_size = 480
image = load_demo_image(image_size=image_size, device=device)

# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
model_url = 'D:/data/ai/BLIP/model_base_vqa_capfilt_large.pth'

model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# question = 'Please list all of the animals and objects and plants in the picture'
question = 'Does the picture have a tree?'

with torch.no_grad():
    answer = model(image, question, train=False, inference='generate')
    print('answer: '+answer[0])
