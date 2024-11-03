from fastapi import FastAPI, Request
import os
import requests
from config import send_config
from data.data_class import InfoPacket

def Mapping_code(name):
    codebook={
        "ShinRamyun": "ramen-1",
        "Chapagetti": "ramen-2",
        "Buldak": "ramen-3",
        "CornChips": "chip-1",
        "Saewookkang": "chip-2",
        "PotatoChips": "chip-3",
        "Powerade": "drink-1",
        "Gatorade": "drink-2",
        "CocaCola": "drink-3",
        "Pepsi": "drink-4"
    }
    return codebook[name] if name in codebook else None


def send_message(data: InfoPacket):
    # 설정 파일 불러오기
    config = send_config()
    config.load_from_json("./data/config/send_config.json")

    # 분류 프로세스에서 받은 데이터
    message = data.message
    count = data.count
    object_name = data.object

    code = Mapping_code(object_name)

    if code is None:
        print("코드 매핑 실패")
        return

    # 변경된 수량을 서버로 전송할 데이터
    send_data = {
        "productCode": code,
        "quantityChange": count
    }

    # 서버로 데이터 전송
    response = requests.patch(config.url + config.patch_dir,
                              json=send_data, headers={config.headers})

    # 전송 성공 여부 확인
    if response.status_code == 200:
        print("데이터 전송 성공")
    else:
        print(f"데이터 전송 실패: {response.status_code}")
        return
