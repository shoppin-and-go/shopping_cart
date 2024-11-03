import cv2
import numpy as np
import os
from data.data_class import ImageInfoPacket
import glob
from config import checkP_config


def save_all_images(frames, save_path, name):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, frame in enumerate(frames):
        try:
            cv2.imwrite(os.path.join(save_path, f"{name}_{i}.jpg"), frame)
        except Exception as e:
            print(f"Error: {e}")
            continue

def save_one_image(frame, save_path, name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(os.path.join(save_path, f"{name}.jpg"), frame)


def erase_file(path):
    for file_path in glob.glob(os.path.join(path, '*')):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")


def get_back(frames):
    print("배경 이미지 추출 중...")
    back = np.mean(frames, axis=0).astype(np.uint8)
    return back


def remove_back(image, back, threshold):
    diff = cv2.absdiff(image, back)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(image, mask_3ch)

    return result


def remove_back_white(image, back, threshold):
    diff = cv2.absdiff(image, back)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    mask_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    white_background = np.ones_like(image, dtype=np.uint8) * 255
    result_background = cv2.bitwise_and(white_background, mask_3ch)
    foreground = cv2.bitwise_and(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    result = cv2.add(foreground, result_background)

    return result


def check_process(input_queue, output_queue):
    print("Checking process...")

    config = checkP_config()
    config.load_from_json("./data/config/checkP_config.json")

    output_queue.put("ready")
    save_path = config.save_dir

    erase_file(save_path)
    erase_file(config.save_back_dir)
    erase_file(config.save_diff_dir)

    while True:
        data: ImageInfoPacket = input_queue.get()
        if data.message == "stop":
            print("체크 포르세스에서 종료 신호 수신")
            break
        else:
            print(f"Message: {data.message}")
            process = data.name
            print(f"Process: {process}")

            images = data.image_list
            L = len(images)
            print(f"Number of images: {len(images)}")

            if len(images) < config.min_frame:
                print("이미지가 충분하지 않습니다.")
                continue


            save_all_images(images, save_path, process)
            print("Images saved.")


            back = get_back(images[:config.back])
            save_one_image(back, config.save_back_dir, f"back_{process}")
            print("Back image saved.")

            p1 = []
            for i in range(config.start_frame, config.start_frame + config.check_num):
                frame = images[i]
                result = remove_back_white(frame, back, config.threshold)
                p1.append(result)

            save_all_images(p1, config.save_diff_dir, f"p{process}_1")

            p2 = []
            back2 = images[-1]
            save_one_image(back2, config.save_back_dir, f"back2_{process}")
            for i in range(L-config.check_num-5, L-5):
                frame = images[i]
                result = remove_back_white(frame, back2, config.threshold)
                p2.append(result)

            save_all_images(p2, config.save_diff_dir, f"p{process}_2")

            IIP = ImageInfoPacket(message="check", name=process, image_list=p1+p2)
            output_queue.put(IIP)

    print("check_process.py 종료")
