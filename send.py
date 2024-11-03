from fastapi import FastAPI, Request
import os
import requests
from config import send_config
from data.data_class import InfoPacket


def send_message(data: InfoPacket):
    # 설정 파일 불러오기
    config = send_config()
    config.load_from_json("./data/config/send_config.json")

    # 분류 프로세스에서 받은 데이터
    message = data.message
    count = data.count
    object_name = data.object

    # 기존의 수량을 읽음
    response1 = requests.get(config.url + config.get_dir)

    if response1.status_code == 200:
        print("데이터 수신 성공")

        try:
            # json 파일로 읽음
            response1_json = response1.json()
            # result > inventory > items 안에 있는 데이터를 읽음
            items = response1_json.get("result", {}).get("inventory", {}).get("items", [])

            # 아이템 이름이 일치하면 수량을 변경
            new_quantity = count  # for문 안에서 없을 경우 처리하기 위해 미리 선언
            for item in items:
                if item.get("name") == object_name:
                    original_quantity = item.get("quantity", 0) # 기존 수량
                    new_quantity = original_quantity + count # new_quantity = 기존 수량 + 변경할 수량 으로 변경
                    print(f"아이템 {object_name}의 수량이 {original_quantity}에서 {new_quantity}로 변경됩니다.")
                    break
                else:
                    print("데이터 처리 실패")
                    return
        except:
            print("데이터 처리 실패")
            return

    else:
        print(f"데이터 수신 실패: {response1.status_code}")
        return

    # 변경된 수량을 서버로 전송할 데이터
    send_data = {
        "productCode": object_name,
        "quantityChange": new_quantity
    }

    # 서버로 데이터 전송
    response2 = requests.patch(config.url + config.patch_dir,
                               json=send_data, headers={config.headers})

    # 전송 성공 여부 확인
    if response2.status_code == 200:
        print("데이터 전송 성공")
    else:
        print(f"데이터 전송 실패: {response2.status_code}")
        return
