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
        print(f"{ip}:{port} 서버에 연결되었습니다.")
    except Exception as e:
        print(f"{ip}:{port} 서버 연결 오류 - {e}")
        sock = None
    return sock

def send_frame(sock, cam):
    while True:
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
        # time.sleep(1/30)

def send_status(sock_central):
    while True:
        try:
            # 장치 상태 확인
            arduino_avail = os.path.exists('/dev/ttyArduino')
            vid0_avail = os.path.exists('/dev/video0')
            vid1_avail = os.path.exists('/dev/video1')
            status = 1 if arduino_avail and vid0_avail and vid1_avail else 0

            status_packet = b'CS' + status.to_bytes(1, byteorder="big")
            sock_central.sendall(status_packet + b'\n')
            print(f"상태 전송: CS {status}")
        except Exception as e:
            print(f"상태 전송 오류: {e}")
        time.sleep(5)  # 5초마다 상태 전송

def recieve_motor(sock_central, ser):
    buffer = b''
    while True:
        try:
            data = sock_central.recv(1024)
            if not data:
                print("서버로부터 연결이 종료되었습니다.")
                break
            buffer += data
            while b'\n' in buffer:
                message, buffer = buffer.split(b'\n', 1)
                if not message:
                    continue
                # 메시지 형식: 시작 바이트 + 왼쪽 값 + 오른쪽 값
                if len(message) >= 3:
                    start_byte = int.from_bytes(message[:1], byteorder="big")
                    print(start_byte)
                    if start_byte == 10:  # 시작 바이트가 10인지 확인
                        left_value = message[1]
                        right_value = message[2]

                        print(f"모터 값: {left_value}, {right_value}")

                        cmd = message[1:3]
                        ser.write(cmd)
                        print(f"시리얼로 전송: {cmd}")

                        # 아두이노로부터 응답 읽기
                        if ser.in_waiting > 0:
                            ar_msg = ser.read(ser.in_waiting)
                            print(f"시리얼로부터 수신: {ar_msg}")
                    else:
                        print(f"알 수 없는 시작 바이트: {start_byte}")
                else:
                    print("불완전한 모터 명령 수신")
        except Exception as e:
            print(f"모터 명령 수신 오류: {e}")
            break

def main():

    # 카메라 초기화 (필요한 경우)
    cam0 = cv2.VideoCapture(0)
    # cam1 = cv2.VideoCapture(1)

    # 중앙 서버에 연결
    central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    if not central_sock:
        return  # 연결 실패 시 종료

    # 아두이노 연결
    try:
        ser = serial.Serial('/dev/ttyArduino', 9600)
        print("아두이노에 연결되었습니다.")
    except serial.SerialException as e:
        print(f"시리얼 포트 열기 오류: {e}")
        ser = None

    # 스레드 생성
    frame0_thread = threading.Thread(target=send_frame, args=(central_sock, cam0))
    status_thread = threading.Thread(target=send_status, args=(central_sock,))
    if ser:
        motor_command_thread = threading.Thread(target=recieve_motor, args=(central_sock, ser))
    else:
        motor_command_thread = None

    # 스레드 시작
    frame0_thread.start()
    status_thread.start()
    if motor_command_thread:
        motor_command_thread.start()

    try:
        frame0_thread.join()
        status_thread.join()
        if motor_command_thread:
            motor_command_thread.join()
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")
    finally:
        # 리소스 정리
        cam0.release()
        if central_sock:
            central_sock.close()
        if ser:
            ser.close()

if __name__ == "__main__":
    main()



