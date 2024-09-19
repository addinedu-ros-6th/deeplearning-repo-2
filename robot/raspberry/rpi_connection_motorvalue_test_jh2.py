import cv2
import socket
import struct
import os
import threading
import time
import serial

# 중앙 서버 정보
CENTRAL_SERVER_IP = "192.168.0.147"
CENTRAL_SERVER_PORT = 3141

def connect_to_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((ip, port))
        print(f"{ip}:{port}에 연결되었습니다.")
    except Exception as e:
        print(f"{ip}:{port} 서버에 연결 오류 - {e}")
        sock = None
    return sock

def send_frame(sock, cam):
    while True:
        try:
            ret, frame = cam.read()
            if ret and sock:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                frame_size = len(frame_data)
                header = b'SF'
                sock.sendall(header + struct.pack(">L", frame_size) + frame_data + b'\n')
                print(f"{sock.getpeername()}에 {frame_size} 바이트의 프레임 전송")
        except Exception as e:
            print(f"{cam} 프레임 캡처 또는 전송 오류: {e}")
        # time.sleep(1/30)

def send_status(sock_central):
    while True:
        try:
            arduino_avail = os.path.exists('/dev/ttyACM0')
            vid0_avail = os.path.exists('/dev/video0')
            vid1_avail = os.path.exists('/dev/video1')
            status = 1 if arduino_avail and vid0_avail and vid1_avail else 0
        
            status_packet = b'CS' + status.to_bytes(1, byteorder="big")
            sock_central.sendall(status_packet + b'\n')
            print(f"상태 전송: CS {status}")
        except Exception as e:
            print(f"상태 전송 오류: {e}")
        time.sleep(5)

def receive_motor(sock_central, ser):
    while True:
        msg = b''
        while len(msg) < 4:
            data = sock_central.recv(1)
            if not data:
                # 연결이 끊어진 경우
                return
            msg += data

        if msg[0] == 10 and msg[3] == ord('\n'):
            left_value = msg[1]
            right_value = msg[2]

            print(f"모터 명령 수신: 왼쪽={left_value}, 오른쪽={right_value}")

            # 아두이노로 데이터 전송
            ser.write(msg[:3])
            print(f"아두이노로 전송: {msg[:3]}")

            # 아두이노로부터 응답 읽기
            if ser.in_waiting > 0:
                ar_msg = ser.read(ser.in_waiting)
                print(f"아두이노로부터 수신: {ar_msg}")
        else:
            print("유효하지 않은 모터 명령 수신.")

def test_arduino(ser):
    cmd = bytes([10, 128, 128])
    ser.write(cmd)
    print(f"아두이노로 전송: {cmd}")

def main():
    # 카메라 초기화 (필요한 경우)
    # cam0 = cv2.VideoCapture(0)
    # cam1 = cv2.VideoCapture(1)

    # 중앙 서버 연결
    central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    if not central_sock:
        return

    # 아두이노 연결
    ser = serial.Serial('/dev/ttyACM0', 9600)

    # 아두이노 통신 테스트
    test_arduino(ser)

    # 스레드 생성
    # frame0_thread = threading.Thread(target=send_frame, args=(central_sock, cam0))
    status_thread = threading.Thread(target=send_status, args=(central_sock,))
    motor_command_thread = threading.Thread(target=receive_motor, args=(central_sock, ser))

    # 스레드 시작
    # frame0_thread.start()
    status_thread.start()
    motor_command_thread.start()

    try:
        # frame0_thread.join()
        status_thread.join()
        motor_command_thread.join()
    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")

if __name__ == "__main__":
    main()

