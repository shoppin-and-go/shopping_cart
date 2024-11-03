from classification import classification_process
from detect import read_video
from send import send_message
from process_check import check_process
import multiprocessing
from PIL import Image
import os
import time
from data.data_class import ImageInfoPacket
import sys
import threading
import queue


def keyboard_listener(stop_event, input_queue_list, stopSig):
    while not stop_event.is_set():
        key = input()
        if key == 'q':
            print("프로그램 종료 명령을 받았습니다.")
            for q in input_queue_list:
                q.put(stopSig)
            stop_event.set()


if __name__ == "__main__":
    c_input_queue = multiprocessing.Queue()
    c_output_queue = multiprocessing.Queue()
    d_input_queue = multiprocessing.Queue()
    d_output_queue = multiprocessing.Queue()
    p_input_queue = multiprocessing.Queue()
    p_output_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=classification_process, args=(c_input_queue, c_output_queue))
    p2 = multiprocessing.Process(target=read_video, args=(d_input_queue, d_output_queue))
    p3 = multiprocessing.Process(target=check_process, args=(p_input_queue, p_output_queue))

    p1.start()
    message = c_output_queue.get()
    if message == "ready":
        print("모델 로딩 완료.")

    p3.start()
    message = p_output_queue.get()
    if message == "ready":
        print("프로세스 체크 시작")

    p2.start()
    message = d_output_queue.get()
    if message == "ready":
        print("비디오 읽기 시작")

    stopSig = ImageInfoPacket(message="stop", name="stop", image_list=[])

    stop_event = threading.Event()
    input_queue_list = [c_input_queue, d_input_queue, p_input_queue]
    stop_thread = threading.Thread(target=keyboard_listener, args=(stop_event, input_queue_list, stopSig))
    stop_thread.start()

    try:
        while not stop_event.is_set():
            try:
                image = d_output_queue.get(timeout=1)
                print("이미지 전달")
                if image is not None:
                    p_input_queue.put(image)
                    try:
                        result = p_output_queue.get(timeout=3)
                        if result:
                            print("결과 전달")
                            c_input_queue.put(result)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue

            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            except EOFError:
                break

            except KeyboardInterrupt:
                print("프로그램 종료")
                for q in input_queue_list:
                    q.put(stopSig)
                stop_event.set()
                break

    finally:
        stop_thread.join()
        p1.join()
        p2.join()
        p3.join()
        print("메인 프로세스 종료")
        sys.exit(0)
