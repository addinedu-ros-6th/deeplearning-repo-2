## ----------- note -----------
## 9월 15-16일 기준 너무너무 잘됨.
## 대신 영상 받아오는 방식이 살짝 바껴서 서버 코드는 제가 16일날 슬랙에 올려둔걸로 써야 잘돌아갈것임...
## 현재 아두이노는 0915_final_arduino 파일이 컴파일, 업로드 되어 있는 상태

## 라즈베리파이에서 central server로 실시간 영상 프레임 보내고 10초에 한번씩 로봇 연결 상태 보내는 코드.
## 주행뷰 광각캠과 로봇 연결상태(CS)는 central server로, 꽃뷰 웹캠은 pollination server로 보내지는 코드.
## 메인 서버에서 받은 로봇방향(MR, ML, MF, MS)과 왼쪽 오른쪽 모터값을 받고 아두이노로 보내는 코드.
## 미완성인 부분: (GUI에서 받고) 메인 서버에서 넘겨받은 로봇 스캔 시간 설정이나 (SS) 로봇 제어 명령 (RC, R) or (RC, S)에 따라 움직이는 코드.
## -----------------------------

import cv2
import socket
import struct
import os
import time
import threading
import serial

# Central server details
CENTRAL_SERVER_IP = '192.168.0.13'  # Central IP
CENTRAL_SERVER_PORT = 3141  # Central port

# Pollination server details (if needed)
POLLINATION_SERVER_IP = '192.168.0.42'  # Pollination IP
POLLINATION_SERVER_PORT = 9003  # Pollination port

def connect_to_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((ip, port))
        print(f"Connected to server at {ip}:{port}")
    except Exception as e:
        print(f"Error connecting to server {ip}:{port} - {e}")
        sock = None
    return sock

def send_status(sock_central):
    while True:
        try:
            # Check for device availability
            arduino_avail = os.path.exists('/dev/ttyArduino')
            vid0_avail = os.path.exists('/dev/video0')
            vid1_avail = os.path.exists('/dev/video1')
            status = 1 if arduino_avail and vid0_avail and vid1_avail else 0
        
            status_packet = struct.pack('2sh', b'CS', status)
            sock_central.sendall(status_packet + b'\n')
            print(f"Sent status: CS {status}")
        except Exception as e:
            print(f"Error sending status: {e}")
        time.sleep(1)  # Send status every 10 seconds

def send_frame(sock, cam):
    while True:
        try:
            # Capture frame from /dev/video0 or /dev/video1
            ret, frame = cam.read()
            if ret and sock:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                frame_size = len(frame_data) # Send size and frame data
                header = b'SF'
                sock.sendall(header + struct.pack(">L", frame_size) + frame_data + b'\n')
                print(f"Sent frame of size {frame_size} to {sock.getpeername()}")
        except Exception as e:
            print(f"Error capturing or sending {cam} frame: {e}")
        # Limit to 30 frames per second
        time.sleep(1/30)

def receive_motor_cmd(sock, ser):
    while True:
        try:
            # Receive command ID (2 bytes) + left speed (2 bytes) + right speed (2 bytes)
            data = sock.recv(7)
            if data.endswith(b'\n'):
                data = data.strip()
                if len(data) == 6 and data.startswith(b'M'): #as the command id will be MR, ML, MF, MS
                    # Unpack the data
                    command_id = data[:2].decode('utf-8')
                    left_speed = struct.unpack('h', data[2:4])[0]
                    right_speed = struct.unpack('h', data[4:6])[0]

                    print(f"Received command: {command_id}, left speed: {left_speed}, right speed: {right_speed}")

                    # Format the command for the Arduino
                    cmd = f'{command_id}{left_speed},{right_speed}\n'.encode()
                    ser.write(cmd)

        except Exception as e:
            print(f"Error receiving or processing motor commands: {e}")

# ---------------------------------------------------------------------------------------
# this portion will be changed once the GUI and the aruco marker portion are completed.
def receive_control_cmds(sock, ser):
    while True:
        # try:
            # Receive control commands
            data = sock.recv(4)  # expecting 4 byte data but this can be changed later
            if data.endswith(b'\n'):
                data = data.strip()
                if data:
                    # Check for scanning schedule command
                    if data.startswith(b'SS'):
                        time_to_scan = struct.unpack('h', data[2:4])[0]  # ex) 15 stands for 3pm and 08 for 8am
                        print(f"Received scanning schedule command: Set scan at {time_to_scan} hours")
                        # Send a command to Arduino to start patrolling the orchard
                        # once the whole map is drawn, will make a patrol route for the robot to move along the lanes
                        # ser.write(b'P\n')  # Example command to start patrol

                    # Check for robot control command
                    elif data.startswith(b'RC'):
                        control_action = data[2:3].decode('utf-8')
                        if control_action == 'R':
                            print("Received robot control RETURN cmd: Return to WS") #Work Station
                            # aruco marker detection will be placed here
                            #ser.write(b'MS\n')  # this is an example
                        elif control_action == 'S':
                            print("Received robot control START cmd: Start Patrol")
                            # once the whole map is drawn, will make a patrol route for the robot to move along the lanes
                            #ser.write(b'MF\n')  # this is an example

        # except Exception as e:
        #     print(f"Error receiving or processing control commands: {e}")
# --------------------------------------------------------------------------------------------

def main():
    # Initialize cameras
    cam0 = cv2.VideoCapture(0)  # fyi, the index number changes to 2 sometimes
    cam1 = cv2.VideoCapture(1)  # fyi, the index number changes to 0 sometimes

    # Connect to central server
    central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    if not central_sock:
        return  # Exit if connection fails

    # Connect to pollination server
    pollination_sock = connect_to_server(POLLINATION_SERVER_IP, POLLINATION_SERVER_PORT)
    if not pollination_sock:
        return  # Exit if connection fails

    # Initialize serial connection to Arduino for precise motor control
    try:
        ser = serial.Serial('/dev/ttyArduino', 9600)
    except Exception as e:
        print(f"Error opening serial port to Arduino: {e}")
        return

    # Create threads
    frame0_thread = threading.Thread(target=send_frame, args=(central_sock, cam0))
    frame1_thread = threading.Thread(target=send_frame, args=(pollination_sock, cam1))
    status_thread = threading.Thread(target=send_status, args=(central_sock,))
    # motor_command_thread = threading.Thread(target=receive_motor_cmd, args=(central_sock, ser))
    # control_command_thread = threading.Thread(target=receive_control_cmds, args=(central_sock, ser))

    # Start threads
    frame0_thread.start()
    frame1_thread.start()
    status_thread.start()
    # motor_command_thread.start()
    # control_command_thread.start()

    try:
        # Wait for threads to finish
        frame0_thread.join()
        frame1_thread.join()
        status_thread.join()
        # motor_command_thread.join()
        # control_command_thread.join()


    except KeyboardInterrupt:
        ser.write(b'MS\n')  # rmb to send 'MS' cmd to the arduino serial port
        print("Interrupted by user")
        
    finally:
        cam0.release()
        cam1.release()
        
        if central_sock:
            central_sock.close()
        if pollination_sock:
            pollination_sock.close()

        if ser:
            ser.close()

if __name__ == '__main__':
    main()

