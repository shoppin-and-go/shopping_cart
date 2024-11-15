import cv2
import numpy as np
import os
from dataclasses import dataclass
import base64
import json
from config import detect_config
from data.data_class import ImageInfoPacket
import queue
import sys


@dataclass
class Image:
    name: str
    image: np.ndarray
    save_path: str


def get_width_height_area(x, y, w, h):
    width = w - x
    height = h - y
    area = width * height
    return width, height, area


# 이미지 저장
def write_image(img: Image):
    pass
    if not os.path.exists(img.save_path):
        os.makedirs(img.save_path)
    img_name = os.path.join(img.save_path, img.name)
    cv2.imwrite(img_name, img.image)


# cv2로 읽은 이미지를 base64로 인코딩
def encode_image(img: Image):
    _, img_bytes = cv2.imencode('.jpg', img.image)
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_image


# 이미지 노이즈 제거
def denoise_thresholded_image(kernel, img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    fg_threshold = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_threshold = cv2.dilate(fg_threshold, kernel, iterations=2)
    return fg_threshold


# 이미지에서 ROI 영역만 자르기
def cut_roi(frame, roi):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = roi
    output = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return output


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def read_video(input_queue, output_queue):
    print("비디오 읽는 중...")

    # 설정 파일 읽기
    config = detect_config()
    config.load_from_json(resource_path("./data/config/detect_config.json"))

    # 비디오 파일 읽기
    video = config.video
    capture = cv2.VideoCapture(0)

    # FPS 확인
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    if not capture.set(cv2.CAP_PROP_FPS, config.fps):
        print("FPS 설정 실패")
    else:
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        print(f"설정된 FPS: {actual_fps}")

    # 배경 구하기
    back = cv2.createBackgroundSubtractorMOG2(history=config.backHist, varThreshold=config.backThresh,
                                              detectShadows=True)

    # 움직임 감지할 ROI 설정
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    D_roi_top_left_x = int(frame_width * config.detect_ROI[0])
    D_roi_top_left_y = int(frame_height * config.detect_ROI[1])
    D_roi_bottom_right_x = int(frame_width * config.detect_ROI[2])
    D_roi_bottom_right_y = int(frame_height * config.detect_ROI[3])
    D_roi_w, D_roi_h, D_roi_area = get_width_height_area(D_roi_top_left_x, D_roi_top_left_y, D_roi_bottom_right_x,
                                                         D_roi_bottom_right_y)

    # 캡쳐할 ROI 설정
    C_roi_top_left_x = int(frame_width * config.capture_ROI[0])
    C_roi_top_left_y = int(frame_height * config.capture_ROI[1])
    C_roi_bottom_right_x = int(frame_width * config.capture_ROI[2])
    C_roi_bottom_right_y = int(frame_height * config.capture_ROI[3])
    capture_ROI = (C_roi_top_left_x, C_roi_top_left_y, C_roi_bottom_right_x, C_roi_bottom_right_y)

    # 각종 변수 초기화
    frame_counter = 0  # 현재 프레임이 몇 번째 프레임인지 카운트
    process_counter = 1  # 넣고 빼는 과정이 몇 번 있었는지 카운트
    state = "WAITING"  # 현재 상태
    before_state = "WAITING"  # 이전 상태
    update_state = False  # 상태 업데이트 여부

    sendingMode = False  # 전송 모드
    tempFrames = []  # 이전 프레임 임시 저장
    transferFrames = []  # 전송할 프레임 저장
    MAX_TEMP_FRAMES = 5

    output_queue.put("ready")

    while True:
        try:
            msg = input_queue.get_nowait()
            if isinstance(msg, ImageInfoPacket) and msg.message == "stop":
                print("감지 프로세스에서 종료 신호 수신")
                break
        except queue.Empty:
            pass

        ret, frame = capture.read()
        if not ret:  # 읽기 실패 혹은 더 이상 읽을 비디오가 없을 때
            print("비디오 읽기 실패 또는 비디오 종료")
            break

        # 원본 프레임 저장
        original_frame = frame.copy()
        frame_counter += 1

        # 모든 프레임을 tempFrames에 추가
        tempFrames.append(original_frame)

        # tempFrames의 크기 제한
        if len(tempFrames) > MAX_TEMP_FRAMES:
            tempFrames.pop(0)

        # 프레임 스킵
        if capture.get(cv2.CAP_PROP_POS_FRAMES) % config.frames_skip != 0:
            continue

        # ROI 추출 -> D_roi: 움직임 감지할 ROI, C_roi: 캡쳐할 ROI
        D_roi = frame[D_roi_top_left_y:D_roi_bottom_right_y, D_roi_top_left_x:D_roi_bottom_right_x]
        C_roi = frame[C_roi_top_left_y:C_roi_bottom_right_y, C_roi_top_left_x:C_roi_bottom_right_x]

        # 배경 추출 및 전경 마스크 생성
        foreground_mask = back.apply(D_roi)
        _, fg_threshold = cv2.threshold(foreground_mask, 254, 255, cv2.THRESH_BINARY)

        # 노이즈 제거를 위한 모폴로지 연산
        kernel = (3, 3)
        fg_threshold = denoise_thresholded_image(kernel, fg_threshold)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(fg_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 물체 감지
        if len(contours) > 0:
            # 면적이 가장 큰 것만 선택
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > config.detectPercents * D_roi_area:  # 물체기 ROI 면적의 일정 퍼센트 이상이면
                before_state = state
                state = "DETECTED"
            else:
                before_state = state
                state = "Not Enough Area"
        else:
            before_state = state
            state = "Not Detected"

        # 물체가 감지된 경우
        if state == "DETECTED":
            if not sendingMode:
                print("")
                print(f"오브젝트 감지 Process {process_counter}")
                sendingMode = True
                print(f"tempFrames: {len(tempFrames)}")
                transferFrames = tempFrames.copy()
                tempFrames = []
            transferFrames.append(original_frame)
        elif state == "Not Detected":
            if sendingMode:
                transferFrames.append(original_frame)
                print(f"{process_counter}_오브젝트 검출 프레임 수: {len(transferFrames)}")
                sendingMode = False
                cut_transferFrames = []
                for f in transferFrames:
                    cut_transferFrames.append(cut_roi(f, capture_ROI))
                IIP = ImageInfoPacket(message=f"{frame_counter}", name=f"{process_counter}", image_list=cut_transferFrames)
                output_queue.put(IIP)
                transferFrames = []
                process_counter += 1

        # ROI 영역을 사각형으로 표시
        cv2.rectangle(frame, (D_roi_top_left_x, D_roi_top_left_y), (D_roi_bottom_right_x, D_roi_bottom_right_y),
                      (255, 0, 0), 2)
        cv2.rectangle(frame, (C_roi_top_left_x, C_roi_top_left_y), (C_roi_bottom_right_x, C_roi_bottom_right_y),
                      (0, 0, 255), 2)

        # cv2를 통해 감지되는 부분 표시
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", fg_threshold)

        if cv2.waitKey(50) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    print("detect.py 종료")
