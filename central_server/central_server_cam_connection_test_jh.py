## ----------- note -----------
## 9월 15-16일 기준 라즈베리 파이에서 영상, CS 받아오는거 딜레이 없이 꽤 잘됨.
## 모터값 쏘는 부분은 오른쪽 바퀴 사망이슈로 인해 패킷 살짝 변경됨.
## 그냥 모터값만 주는게 아니라 방향 제어값(MF, MR, ML, MS)이랑 왼쪽 모터값, 오른쪽 모터값 따로 주는걸로. 모터값은 이왕이면 오른쪽이 더 높게 40, 47 추천.
## 영상 수신받는 부분 살짝 변경됨 원래는 pickle 사용해서 프레임 직렬화 시켜서 송신했지만 피클은 딜레이가 심해져서 jpeg형태를 image encode해서 송신하는 방법으로 변경됐기에 서버도 image decode하는 방법으로 변경.
## -----------------------------

import numpy as np
import socket
import struct
import cv2
import threading
import time
import queue

# Global queue for frames
frame_queue = queue.Queue(maxsize=10)  # Limit queue size

def handle_frame(conn):
    try:
        # Receive frame size
        size_data = b''
        while len(size_data) < 4:
            packet = conn.recv(4 - len(size_data))
            if not packet:
                return
            size_data += packet
        
        frame_size = struct.unpack('>L', size_data)[0]

        # Receive frame data
        frame_data = b''
        while len(frame_data) < frame_size:
            packet = conn.recv(frame_size - len(frame_data))
            if not packet:
                return
            frame_data += packet
        
        conn.recv(1) # receive and discard the '\n' end byte

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Use a queue with a size limit to manage frame buffer
        if frame_queue.full():
            frame_queue.get()  # Discard oldest frame if queue is full
        frame_queue.put(frame)

        # Display the frame
        while not frame_queue.empty():
            frame_to_display = frame_queue.get()
            cv2.imshow("Received Frame", frame_to_display)
            cv2.waitKey(1)

    except Exception as e:
        print(f"Error handling frame: {e}")


def handle_client(conn):
    try:
        while True:
            # Receive header
            header = b''
            while len(header) < 2:
                packet = conn.recv(2 - len(header))
                if not packet:
                    return
                header += packet

            if header == b'SF':  # Video frame
                handle_frame(conn)
            else:
                print(f"Unknown header received: {header}")
                # Attempt to resynchronize
                conn.recv(1024)  # 싱크 맞추기 위해 쓰레기값 버리기

    except Exception as e:
        print(f"Error handling client: {e}")

if __name__ == '__main__':
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(('192.168.0.13', 3141))
    server_sock.listen(1)

    print(f"Server listening on 192.168.0.13 : 3141. Waiting for client")

    conn, addr = server_sock.accept()
    print(f"Connection from {addr}")

    client_thread = threading.Thread(target=handle_client, args=(conn,))
    client_thread.start()

    conn.close()
    server_sock.close()
    cv2.destroyAllWindows()