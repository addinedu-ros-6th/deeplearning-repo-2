import socket
import struct
import pickle
import cv2
import multiprocessing
import time

def receive_stream(conn, window_name):
    data = b""
    payload_size = struct.calcsize(">L")
    prev_time = time.time()

    while True:
        try:
            # 패킷 헤더(프레임 길이) 수신
            while len(data) < payload_size:
                data += conn.recv(4096)
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # 실제 프레임 데이터 수신
            while len(data) < msg_size:
                data += conn.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 프레임 디코딩
            frame = pickle.loads(frame_data)

    
            # 수신한 프레임을 화면에 표시
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"스트리밍 오류 발생: {e}")
            break

    conn.close()
    cv2.destroyAllWindows()

def start_server(server_ip, server_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(2)

    print(f"서버 시작: {server_ip}:{server_port}, 클라이언트 연결 대기 중...")

    conn, addr = server_socket.accept()
    receive_stream(conn, "Picamera")    

    server_socket.close()

if __name__ == '__main__':
    server_ip = "172.30.1.31" 
    server_port = 12345
    start_server(server_ip, server_port)