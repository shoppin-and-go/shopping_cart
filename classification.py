import torch
from torchvision import transforms
import timm
from PIL import Image
import os
import torch.nn.functional as F
from config import swin_config
import numpy as np
from data.data_class import ImageInfoPacket
from send import send_message


def create_model(config):
    model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=config.num_classes)
    return model


class Classifier:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.model = create_model(config)
        self.model = self.model.to(self.device)
        print(
            f"모델: {config.model_name}, 총 파라미터 수: {sum(p.numel() for p in self.model.parameters())}, 디바이스: {self.device}")

        best_model_path_file = os.path.join(config.model_dir, "best_model.txt")
        with open(best_model_path_file, "r") as f:
            best_model = f.read().strip()
            print(f"최고 모델 로드 중: {best_model}")

        self.model.load_state_dict(torch.load(best_model, map_location=self.device))
        self.model.eval()

        self.data_transforms = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])

    def classify_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("입력은 PIL 이미지 또는 NumPy 배열이어야 합니다.")

        input_tensor = self.data_transforms(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]

            class_probs = {}
            for i in range(self.config.num_classes):
                if i in self.config.idx_to_class:
                    class_probs[self.config.idx_to_class[i]] = round(float(probs[i]), 2)
                else:
                    print(f"키 {i}가 idx_to_class에 존재하지 않습니다.")
                    class_probs[f"Unknown_{i}"] = round(float(probs[i]), 2)

        return class_probs


def calculate_max_prob(prob_list):
    if not prob_list:
        return None, 0

    max_prob = -1
    max_class = None

    for prob_dict in prob_list:
        for cls, prob in prob_dict.items():
            if prob > max_prob:
                max_prob = prob
                max_class = cls

    return max_class, max_prob


def classification_process(input_queue, output_queue):
    print("Classification process started")

    config = swin_config()
    config.load_from_json("./data/config/swin_config.json")
    print(config)
    classifier = Classifier(config)

    print("Ready to classify images")
    output_queue.put("ready")

    while True:
        data: ImageInfoPacket = input_queue.get()
        if data.message == "stop":
            print("분류 프로세스에서 종료 신호 수신")
            break
        else:
            result_list = []
            for image in data.image_list:
                result = classifier.classify_image(image)
                result_list.append(result)

        p1 = []
        p2 = []

        for i, result in enumerate(result_list):
            if i < len(result_list) // 2:
                p1.append(result)
            else:
                p2.append(result)

        max_class1, max_prob1 = calculate_max_prob(p1)
        max_class2, max_prob2 = calculate_max_prob(p2)

        send_message(("Input", max_class1, max_prob1))
        send_message(("Output", max_class2, max_prob2))

    print("classification.py 종료")
