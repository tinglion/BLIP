from models.blip_vqa import blip_vqa
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import argparse
import json

image_size = 480
model_url = "D:/data/ai/BLIP/model_base_vqa_capfilt_large.pth"
path_names = "map.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = blip_vqa(pretrained=model_url, image_size=image_size, vit="base")
model.eval()
model = model.to(device)

with open(path_names, "r", encoding="utf-8") as fp:
    name_map = json.load(fp)


def load_demo_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert("RGB")

    w, h = raw_image.size
    # display(raw_image.resize((w//5, h//5)))

    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def processQ(model, image, question, question_cn, print_q=False):
    if print_q:
        print(f"问: {question_cn}")
    answer = model(image, question, train=False, inference="generate")
    if print_q:
        print("答: " + ("是" if answer[0] == "yes" else "否"))
    return answer[0] == "yes"


def process(model, image_path):
    image = load_demo_image(image_path, image_size=image_size, device=device)
    objs = []

    print("标签提取中，包含：", end="", flush=True)

    for name_en in name_map["ip"]:
        name_cn = name_map["ip"][name_en]
        ret = processQ(
            model,
            image=image,
            # question=f"Does the picture have a {name_en}?",
            question=f"Is a {name_en} in the image?",
            question_cn=f"图片中是否有{name_cn}？",
        )
        if ret:
            print(name_cn, end=" ", flush=True)
            objs.append(name_cn)

    for name_en in name_map["scene"]:
        name_cn = name_map["scene"][name_en]
        ret = processQ(
            model,
            image=image,
            question=f"Is the image the scene of {name_en}?",
            question_cn=f"图片中场景是{name_cn}么？",
        )
        if ret:
            print(name_cn, end=" ", flush=True)
            objs.append(name_cn)

    for name_en in name_map["object"]:
        name_cn = name_map["object"][name_en]
        ret = processQ(
            model,
            image=image,
            question=f"Is a {name_en} in the image?",
            question_cn=f"图片中是否有{name_cn}？",
        )
        if ret:
            print(name_cn, end=" ", flush=True)
            objs.append(name_cn)

    print(f"\n标签提取完成！\n图片({image_path})里面有：" + "、".join(objs))

    # ret = processQ(
    #     model,
    #     image=image,
    #     question=f"Which of the following objects are included in the picture: the moon, sun, and stars?",
    #     question_cn=f"图片中包含下列哪些物体：月亮、太阳、星星？",
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="叫叫IP识别大模型")
    parser.add_argument(
        "--image", dest="image", type=str, help="path of image"
    )  # , required=True
    args = parser.parse_args()

    image_path = args.image
    with torch.no_grad():
        while True:
            if not image_path:
                image_path = input("\n请输入图片路径：")
            process(model, image_path=image_path)
            image_path = None
