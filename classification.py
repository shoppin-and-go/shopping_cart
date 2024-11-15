import torch
import os
from config import yolo_config
import numpy as np
from data.data_class import ImageInfoPacket, InfoPacket
from send import send_message
from ultralytics import YOLO
import cv2
import sys


class YOLO_Classifier:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.save_dir = config.save_dir
        self.count = 0

        best_model_path_file = resource_path(os.path.join(config.model_dir, "best_model.txt"))
        with open(best_model_path_file, "r") as f:
            best_model = f.read().strip()
            print(f"최고 모델 로드 중: {best_model}")

        best_model_path = resource_path(best_model)
        self.model = YOLO(best_model_path)

        print(f"모델: {config.model_name}, 디바이스: {self.device}")

    def classify_image(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("입력은 NumPy 배열이어야 합니다.")

        with torch.no_grad():
            results = self.model.predict(image, verbose=False)

            for idx, result in enumerate(results):
                save_path = os.path.join(self.save_dir, f"{self.count}_{idx}.jpg")
                result_img = result.plot()
                cv2.imwrite(save_path, result_img)

            self.count += 1

            class_probs = {}
            for result in results:
                for cls_id, conf in zip(result.boxes.cls, result.boxes.conf):
                    cls_name = result.names[int(cls_id)]
                    if cls_name in class_probs:
                        class_probs[cls_name] = max(class_probs[cls_name], round(float(conf), 2))
                    else:
                        class_probs[cls_name] = round(float(conf), 2)

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

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def classification_process(input_queue, output_queue):
    print("Classification process started")

    config = yolo_config()
    config.load_from_json(resource_path("./data/config/yolo_config.json"))
    print(config)
    classifier = YOLO_Classifier(config)

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

            if max_prob1 > 0.5:
                IP1 = InfoPacket(message="classification", count=1, object=max_class1)
                send_message(IP1)
            else:
                print("분류 결과가 0.5 이상이 아님")
                print(max_prob1, max_class1)

            if max_prob2 > 0.5:
                IP2 = InfoPacket(message="classification", count=-1, object=max_class2)
                send_message(IP2)
            else:
                print("분류 결과가 0.5 이상이 아님")
                print(max_prob2, max_class2)

    print("classification.py 종료")
