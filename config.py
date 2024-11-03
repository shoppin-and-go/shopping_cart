import json
import torch
import os
from dataclasses import dataclass


@dataclass
class detect_config:
    video: int = 0  # 입력할 비디오 파일명
    backHist: int = 50  # 배경을 이전 몇 프레임까지 고려할지
    backThresh: int = 15  # 배경과 전경을 구분할 임계값 0 ~ 16(평균) ~ 255
    frames_skip: int = 5  # 몇 프레임마다 처리할지
    detect_ROI: (float, float, float, float) = (0.15, 0.45, 0.85, 0.6)  # 움직임 감지할 ROI 크기 설정
    capture_ROI: (float, float, float, float) = (0.1, 0.3, 0.9, 0.8)  # 화면 캡쳐할 ROI 크기 설정
    detectPercents: float = 0.01  # ROI 중 몇 퍼센트 이상의 객체를 검출할지
    sendIMG_Nums: int = 3  # 전송할 이미지의 개수
    min_Process_Frames: int = 0  # 프로세스로 취급할 최소 프레임 수
    fps: int = 24  # fps 설정

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)


@dataclass
class swin_config:
    model_name: str = "swinv2_tiny_window16_256"
    model_dir: str = "./data/model"
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 200
    pretrained: bool = False
    class_file: str = "./data/product.txt"
    image_size: (int, int) = (256, 256)
    normalize_mean: (float, float, float) = (0.485, 0.456, 0.406)
    normalize_std: (float, float, float) = (0.229, 0.224, 0.225)
    num_workers: int = 0
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    classes = None
    num_classes: int = None
    class_to_idx = None
    idx_to_class = None

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.model_dir = os.path.join(self.model_dir, self.model_name)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.class_file, "r", encoding='utf-8') as f:
            self.classes = f.read().splitlines()

        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

    def save_to_json(self, path):
        data = self.__dict__.copy()
        data['device'] = str(data['device'])
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            data['device'] = torch.device(data['device'])
            self.__dict__.update(data)

        with open(self.class_file, "r", encoding='utf-8') as f:
            self.classes = f.read().splitlines()

        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}


@dataclass
class checkP_config:
    save_dir: str = "./output/send"
    save_back_dir: str = "./output/back"
    save_diff_dir: str = "./output/diff"
    min_frame: int = 10
    check_num: int = 3
    back: int = 3
    threshold = 30
    start_frame = 5

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)


@dataclass
class send_config:
    url = "http://ec2-3-38-128-6.ap-northeast-2.compute.amazonaws.com:"
    port: int = 80
    headers = {'Content-Type': 'application/json'}
    patch_dir = None
    cartCode: str = "cart-1"

    def __post_init__(self):
        self.url = self.url + str(self.port)
        self.patch_dir = "/carts/" + str(self.cartCode) + "/inventory"

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)


def update_detect_config(path="./data/config/detect_config.json"):
    config = detect_config()
    config.save_to_json(path)
    config.load_from_json(path)
    return config


def update_swin_config(path="./data/config/swin_config.json"):
    config = swin_config()
    config.save_to_json(path)
    config.load_from_json(path)
    return config


def update_checkP_config(path="./data/config/checkP_config.json"):
    config = checkP_config()
    config.save_to_json(path)
    config.load_from_json(path)
    return config


def update_send_config(path="./data/config/send_config.json"):
    config = send_config()
    config.save_to_json(path)
    config.load_from_json(path)
    return config


if __name__ == '__main__':
    c = update_detect_config()
    print(c)

    c2 = update_swin_config()
    print(c2)

    c3 = update_checkP_config()
    print(c3)

    c4 = update_send_config()
    print(c4)
