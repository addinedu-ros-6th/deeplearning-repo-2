import cv2
import socket
import struct
import os
import threading
import time
import serial
import sys

# 중앙 서버 정보
CENTRAL_SERVER_IP = "192.168.0.134"
CENTRAL_SERVER_PORT = 3141

# pollination server
POLLINATION_SERVER_IP = "192.168.0.44"
POLLINATION_SERVER_PORT = 9003

robot_state = 0

def connect_to_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((ip, port))
        print(f"{ip}:{port} 서버에 연결되었습니다.")
    except Exception as e:
        print(f"{ip}:{port} 서버 연결 오류 - {e}")
        sock = None
    return sock

def send_frame(sock, cam):
    global robot_state
    while True:
        if robot_state == 0:
            continue
        try:
            # 프레임 캡처
            ret, frame = cam.read()
            if ret and sock:
                # 프레임 인코딩
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                frame_size = len(frame_data)
                header = b'SF'
                sock.sendall(header + struct.pack(">L", frame_size) + frame_data + b'\n')
                print(f"{sock.getpeername()}로 크기 {frame_size}의 프레임을 전송했습니다.")
        except Exception as e:
            print(f"프레임 캡처 또는 전송 오류: {e}")
        # 프레임 레이트 제한
        time.sleep(1/30)

def send_status(sock_central):
    try:
        while True:
        
            # 장치 상태 확인
            arduino_avail = os.path.exists('/dev/ttyArduino')
            vid0_avail = os.path.exists('/dev/video0')
            vid1_avail = os.path.exists('/dev/video1')
            status = 1 if arduino_avail and vid0_avail and vid1_avail else 0

            status_packet = b'CS' + status.to_bytes(1, byteorder="big")
            sock_central.sendall(status_packet + b'\n')
            print(f"상태 전송: CS {status}")
            time.sleep(5)

    except Exception as e:
            print(f"상태 전송 오류: {e}")

def receive_motor(sock_central, ser):
    global robot_state
    while True:
        try:
            header = b''
            # 서버로부터 바이너리 데이터 수신
            while len(header) < 2:
                header += sock_central.recv(2 - len(header))

            if header == b'MC':
                data = sock_central.recv(3)
                left_value = int.from_bytes(data[:1], byteorder="big")
                right_value = int.from_bytes(data[1:2], byteorder="big")

                print(f"수신된 모터 값: 왼쪽={left_value}, 오른쪽={right_value}")
                start = 60

                ser.write(start.to_bytes(1, byteorder="big" ) + data)
                print(f"아두이노로 전송: {data}")

            elif header == b'RC':
                data = sock_central.recv(2)
                state = int.from_bytes(data[:1], byteorder="big")
                print(f"robot state : {state}")
                robot_state = state
                if robot_state == 0:
                    start = 50
                    motor_value = 0
                    send_data = start.to_bytes(1, byteorder="big") + motor_value.to_bytes(1, byteorder="big")*2 + b'\n'
                    ser.write(send_data)
                    print(f"send arduino : {send_data}")

        except Exception as e:
            print(f"모터 명령 수신 오류: {e}")


def main():
    stop_event = threading.Event()

    # 카메라 초기화 (필요한 경우)
    cam0 = cv2.VideoCapture(0)
    cam1 = cv2.VideoCapture(1)

    # 중앙 서버에 연결
    central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    central_sock.settimeout(None)
    if not central_sock:
        return  # 연결 실패 시 종료

    # Pollination server connect
    pollination_sock = connect_to_server(POLLINATION_SERVER_IP, POLLINATION_SERVER_PORT)
    if not pollination_sock:
        return

    # 아두이노 연결
    try:
        ser = serial.Serial(port='/dev/ttyArduino', baudrate=9600)
        print("아두이노에 연결되었습니다.")
    except serial.SerialException as e:
        print(f"시리얼 포트 열기 오류: {e}")
        return
        

    # 스레드 생성
    frame0_thread = threading.Thread(target=send_frame, args=(central_sock, cam0))
    frame1_thread = threading.Thread(target=send_frame, args=(pollination_sock, cam1))
    status_thread = threading.Thread(target=send_status, args=(central_sock,))
    motor_command_thread = threading.Thread(target=receive_motor, args=(central_sock, ser))

    # 스레드 시작
    frame0_thread.start()
    frame1_thread.start()
    status_thread.start()
    motor_command_thread.start()

    try:
        frame0_thread.join()
        frame1_thread.join()
        status_thread.join()
        motor_command_thread.join()
    except KeyboardInterrupt:
        start = 50
        motor_value = 0
        send_data = start.to_bytes(1, byteorder="big") + motor_value.to_bytes(1, byteorder="big")*2 + b'\n'
        ser.write(send_data)
        print(f"send arduino : {send_data}")
        print("사용자에 의해 중단되었습니다.")
        sys.stdout.flush()
        stop_event.set()
    finally:
        # 리소스 정리
        cam0.release()
        cam1.release()
        if central_sock:
            central_sock.close()
        if pollination_sock:
            pollination_sock.close()
        if ser:
            ser.close()
        sys.stdout.flush()

if __name__ == "__main__":
    main()
