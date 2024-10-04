from datetime import datetime
import cv2
import numpy as np
import socket
import struct
import time
from threading import Thread

# Global variables for motor speed
left_speed = 0
right_speed = 0

# Handle GUI client connection (receives schedule time)
def handle_client_gui(conn, addr):
    print(f"GUI connected by {addr}")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        message = data.decode('utf-8')
        if message.startswith("SS"):
            #schedule_time_str = message.split(":")[1]
            schedule_time_str = message[2:].strip()
            print(f"Received schedule time: {schedule_time_str}")
            schedule_time = datetime.strptime(schedule_time_str, "%Y-%m-%d %H:%M:%S")
            
            # Start a thread to wait until the scheduled time
            Thread(target=wait_until_time, args=(schedule_time,)).start()

        conn.sendall(b"Schedule received")
    conn.close()

def wait_until_time(schedule_time):
    while True:
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S")
        if current_time == schedule_time:
            # Send motor values to Raspberry Pi (robot)
            send_motor_value()
            break
        time.sleep(1)

# Handle Raspberry Pi (robot) connection and motor control
def handle_client_rpi(rpi_conn):
    global left_speed, right_speed
    try:
        while True:
            header = b''
            while len(header) < 2:
                header += rpi_conn.recv(2 - len(header))

            # frame data
            if header == b'SF':
                size_data = b''
                while len(size_data) < 4:
                    packet = rpi_conn.recv(4 - len(size_data))
                    if not packet:
                        break
                    size_data += packet
                frame_size = struct.unpack(">L", size_data)[0]

                frame_data = b''
                while len(frame_data) < frame_size:
                    packet = rpi_conn.recv(frame_size - len(frame_data))
                    if not packet:
                        break
                    frame_data += packet

                rpi_conn.recv(1)  # \n 받기

                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    process_frame(frame)  # Process the frame (lane detection)
                else:
                    print("Error: Unable to decode frame")

            # status data
            elif header == b'CS':
                status_data = b''
                while len(status_data) < 1:
                    packet = rpi_conn.recv(1 - len(status_data))
                    if not packet:
                        break
                    status_data += packet

                rpi_conn.recv(1)  # \n 받기

                status = int.from_bytes(status_data, byteorder="big")
                if status == 1:
                    print("status:", status)
                else:
                    print("status", status)

    except Exception as e:
        print(f"Error receiving or displaying frame: {e}")
    finally:
        rpi_conn.close()
        cv2.destroyAllWindows()

def send_motor_value(rpi_conn):
    global left_speed, right_speed
    while True:
        start = 60
        motor_command = start.to_bytes(1, byteorder="big") + left_speed.to_bytes(1, byteorder="big") + right_speed.to_bytes(1, byteorder="big") + b"\n" 

        try:
            rpi_conn.sendall(motor_command)
            print(f"send motor_command {motor_command}")
            print(f"motor value: {left_speed}, {right_speed}")
        except Exception as e:
            print(f"Error sending motor command: {e}")
        
        time.sleep(1)  # 2~5초에 한번 쏴주는것만으로도 충분함. 타임 슬립없으면 충돌남.
               
# Sobel 연산을 통한 x 또는 y 방향 경계 계산 함수
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)
    l_channel = hls[:, :, 1]

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    else:
        raise ValueError("orient must be 'x' or 'y'")

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

# 색상 정보를 기반으로 특정 범위에 해당하는 색상을 강조하는 함수
def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (vthresh[1] >= v_channel)] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output

# 이미지의 특정 영역(ROI)을 선택하는 함수
def apply_roi(image):
    height = image.shape[0]
    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, int(height * 2/5)),
        (image.shape[1], int(height * 2/5)),
        (image.shape[1], height),
        (0, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi_image = cv2.bitwise_and(image, mask)
    return roi_image

# 카메라 매트릭스와 왜곡 계수
mtx = np.array([[1.15753008e+03, 0.00000000e+00, 6.75382833e+02],
                [0.00000000e+00, 1.15189955e+03, 3.86729350e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.26706898,  0.10305542, -0.00088013,  0.00080643, -0.19574027]])


# Lane detection and frame processing logic
def process_frame(frame):
    global left_speed, right_speed
    # 카메라 캘리브레이션 적용
    undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

     # 차선 검출
    gradx = abs_sobel_thresh(undist_frame, orient='x', thresh_min=20, thresh_max=80)
    grady = abs_sobel_thresh(undist_frame, orient='y', thresh_min=20, thresh_max=80)
                    
     # 색상 임계값 적용
    c_binary = color_threshold(undist_frame, sthresh=(30, 255), vthresh=(120, 255))

    # 최종 이진 이미지 생성
    preprocessImage = np.zeros_like(undist_frame[:, :, 0])
    preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    # ROI 적용
    preprocessImage = apply_roi(preprocessImage)
                    
    # 무게 중심을 기반으로 모터 제어
    midpoint = preprocessImage.shape[1] // 2

    nonzero = preprocessImage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (nonzerox < midpoint)
    right_lane_inds = (nonzerox >= midpoint)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 왼쪽 차선의 무게중심 계산 및 표시
    if len(leftx) > 0 and len(lefty) > 0:
        left_center_x = int(np.mean(leftx))
        left_center_y = int(np.mean(lefty))
        cv2.circle(undist_frame, (left_center_x, left_center_y), 5, (0, 255, 0), -1)
    else:
        left_center_x = None
        left_center_y = None

    # 오른쪽 차선의 무게중심 계산 및 표시
    if len(rightx) > 0 and len(righty) > 0:
        right_center_x = int(np.mean(rightx))
        right_center_y = int(np.mean(righty))
        cv2.circle(undist_frame, (right_center_x, right_center_y), 5, (0, 255, 0), -1)
    else:
        right_center_x = None
        right_center_y = None

    # 두 차선 무게중심의 중앙점 계산 및 표시
    if left_center_x is not None and right_center_x is not None:
        mid_center_x = int((left_center_x + right_center_x) / 2)
        mid_center_y = int((left_center_y + right_center_y) / 2)
        cv2.circle(undist_frame, (mid_center_x, mid_center_y), 5, (255, 0, 0), -1)

        # 이미지 중앙과의 편차 계산
        image_center_x = undist_frame.shape[1] // 2
        deviation = mid_center_x - image_center_x

        # 모터 값 계산 및 전송
        max_speed = 100  # 최대 속도
        base_speed = 30  # 기본 속도
        change_speed = 60
        max_deviation = undist_frame.shape[1] // 2

        normalized_deviation = deviation / max_deviation
        normalized_deviation = np.clip(normalized_deviation, -1, 1)
        
        # left_speed = base_speed - (normalized_deviation * change_speed)
        # right_speed = base_speed + (normalized_deviation * change_speed)

    # 차선 예외처리
    right_speed = 30
    left_speed = 30
    if left_center_x is None:
        right_speed += 20
    elif left_center_x <  50 or right_center_x < 520:
        right_speed += 20
    elif right_center_x is None:
        left_speed += 15
    elif right_center_x > 600 or left_center_x > 130:
        left_speed += 15
        left_speed = int(np.clip(left_speed, 0, max_speed))
        right_speed = int(np.clip(right_speed, 0, max_speed))
    elif mid_center_x < 300:
        left_speed += 10
    elif mid_center_x > 330:
        right_speed += 10
    
    left_speed = int(np.clip(left_speed, 0, max_speed))
    right_speed = int(np.clip(right_speed, 0, max_speed))


    # 화면에 디버그용으로 표시
    lane_overlay = np.zeros_like(undist_frame)
    lane_overlay[preprocessImage == 255] = [0, 0, 255]
    combined = cv2.addWeighted(undist_frame, 0.7, lane_overlay, 1.0, 0)
    cv2.imshow("Lane Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True  # To indicate breaking the loop
    return False



# Server initialization
def start_server():
    server_ip = "192.168.0.134"  # Listen on all interfaces
    gui_port = 1234        # Port for GUI client
    rpi_port = 3141         # Port for Raspberry Pi (robot)

    # Start GUI server
    gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gui_socket.bind((server_ip, gui_port))
    gui_socket.listen(1)
    print(f"Waiting for GUI client on {server_ip}:{gui_port}")

    # Start Raspberry Pi server
    rpi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rpi_socket.bind((server_ip, rpi_port))
    rpi_socket.listen(1)
    print(f"Waiting for Raspberry Pi on {server_ip}:{rpi_port}")

    while True:
        # Accept connections for both GUI and Raspberry Pi
        gui_conn, gui_addr = gui_socket.accept()
        rpi_conn, rpi_addr = rpi_socket.accept()

        # Start threads for handling GUI and Raspberry Pi clients
        gui_thread = Thread(target=handle_client_gui, args=(gui_conn, gui_addr))
        rpi_thread = Thread(target=handle_client_rpi, args=(rpi_conn,))

        gui_thread.start()
        rpi_thread.start()

        # Send motor values to Raspberry Pi in a separate thread
        motor_cmd_thread = Thread(target=send_motor_value, args=(rpi_conn,))
        motor_cmd_thread.start()

        gui_thread.join()
        rpi_thread.join()
        motor_cmd_thread.join()

    gui_socket.close()
    rpi_socket.close()

if __name__ == "__main__":
    start_server()