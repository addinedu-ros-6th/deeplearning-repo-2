import cv2
from cv2 import aruco
import numpy as np
import socket
import struct
import time
from threading import Thread
from datetime import datetime, timedelta


# Sobel 연산을 통한 x 또는 y 방향 경계 계산 함수
def abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(float)
    l_channel = hls[:, :, 1]

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    else:
        raise ValueError("orient must be 'x' or 'y'")

    # scaled_sobel이 0으로 나눠지는 것을 방지
    if np.max(abs_sobel) != 0:
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    else:
        scaled_sobel = np.zeros_like(abs_sobel, dtype=np.uint8)

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

# 색상 정보를 기반으로 특정 범위에 해당하는 색상을 강조하는 함수
def color_threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 초록색 차선 검출
    lower_green = np.array([70, 220, 120])
    upper_green = np.array([85, 255, 160])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 흰색 차선 검출
    lower_white = np.array([45, 0, 240])
    upper_white = np.array([60, 15, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 두 마스크를 합침
    combined_mask = cv2.bitwise_or(green_mask, white_mask)
    binary_output = np.zeros_like(combined_mask)
    binary_output[combined_mask > 0] = 1

    return binary_output

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

# 카메라 매트릭스와 왜곡 계수 (필요에 따라 수정)
mtx = np.array([[1.15753008e+03, 0.00000000e+00, 6.75382833e+02],
                [0.00000000e+00, 1.15189955e+03, 3.86729350e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.26706898,  0.10305542, -0.00088013,  0.00080643, -0.19574027]])

left_speed = 0
right_speed = 0
robot_state = 0

# 이전 프레임의 차선 무게중심 좌표 저장 변수 초기화
prev_left_center_x = None
prev_right_center_x = None

# ARUCO marker의 크기와 초점 거리 (1)
focal_length = mtx[0, 0]
MARKER_SIZE = 0.0195

# ARUCO marker 텍스트를 이미지에 그리는 함수 (2)
def draw_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
              scale=0.6, color=(0, 0, 255), thickness=1):
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    img_h, img_w, _ = img.shape

    # Adjust x and y to ensure text is within the image boundaries
    x = max(0, x)
    y = max(0, y)

    # Clip text to fit within image boundaries
    text_width, text_height = text_size
    if x + text_width > img_w or y + text_height > img_h:
        print(f"Skipped drawing text '{text}' due to boundary issues.")
        return

    cv2.putText(img, text, (x, y + text_height), font, scale, color, thickness, cv2.LINE_AA)

def handle_rpi_client(rpi_conn, obs_conn):
    global left_speed, right_speed, robot_state
    global prev_left_center_x, prev_right_center_x
    try:
        while True:
            if robot_state == 0:
                continue
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

                # ARUCO marker 딕셔너리 생성 및 파라미터 설정 (3)
                marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
                param_markers = aruco.DetectorParameters()

                # ARUCO marker grayscale로 변환 (4)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                marker_corners, marker_IDs, reject = aruco.detectMarkers(
                    gray_frame, marker_dict, parameters=param_markers
                )
                
                # ARUCO marker가 검출되면 (5)
                if marker_corners:
                    total_markers = range(0, marker_IDs.size)
                    for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                        # Draw marker boundaries
                        cv2.polylines(
                            frame, 
                            [corners.astype(np.int32)], 
                            True, 
                            (0, 255, 255), 
                            4, 
                            cv2.LINE_AA
                        )
                        corners = corners.reshape(4, 2).astype(int)
                        top_right = tuple(corners[0])
                        top_left = tuple(corners[1])
                        bottom_right = tuple(corners[2])
                        bottom_left = tuple(corners[3])

                        # Calculate pixel size of the marker
                        pixel_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
                        pixel_height = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
                        pixel_size = (pixel_width + pixel_height) / 2  # Average pixel size

                        # Estimate distance using pinhole camera model
                        distance = (MARKER_SIZE * focal_length) / pixel_size

                        # Determine text position (e.g., top_right)
                        text_position = top_right

                        # Prepare text to display
                        text = f"id: {ids[0]} Dist: {round(distance, 2)}m"
                        draw_text(frame, text, text_position)

                if frame is not None:
                    # 카메라 캘리브레이션 적용
                    undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

                    # 캘리브레이션 적용 카메라 obstacle로 전송
                    _, buffer = cv2.imencode(".jpg", undist_frame)
                    obs_frame_data = buffer.tobytes()
                    obs_frame_size = len(obs_frame_data)
                    obs_header = b'SF'
                    obs_conn.sendall(obs_header + struct.pack(">L", obs_frame_size) + obs_frame_data + b'\n')
                    print(f"obs server로 크기 {obs_frame_size}의 프레임을 전송했습니다.")

                    # 차선 검출
                    gradx = abs_sobel_thresh(undist_frame, orient='x', thresh_min=50, thresh_max=255)
                    grady = abs_sobel_thresh(undist_frame, orient='y', thresh_min=50, thresh_max=255)

                    # 색상 임계값 적용
                    c_binary = color_threshold(undist_frame)

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

                    # 두 차선의 무게중심 간의 거리 계산
                    if left_center_x is not None and right_center_x is not None:
                        lane_distance = abs(right_center_x - left_center_x)
                        lane_distance_threshold = 52  # 임계값 설정 (실험을 통해 조절 필요)

                        # 차선 너비를 영상에 표시 (픽셀 단위)
                        cv2.putText(undist_frame, f"Lane Width: {lane_distance}px", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        if lane_distance < lane_distance_threshold:
                            # 두 차선이 매우 가까워서 하나의 차선만 감지된 것으로 판단
                            print("두 차선의 간격이 좁습니다. 하나의 차선으로 간주합니다.")

                            # 이전 프레임의 차선 위치를 활용하여 어느 차선이 사라졌는지 판단
                            if prev_left_center_x is not None and prev_right_center_x is not None:
                                # 현재 감지된 차선의 위치가 이전 왼쪽 차선 위치와 가까운지, 오른쪽 차선 위치와 가까운지 판단
                                if abs(left_center_x - prev_left_center_x) < abs(left_center_x - prev_right_center_x):
                                    # 오른쪽 차선이 사라진 것으로 판단하고 오른쪽으로 회전
                                    print("오른쪽 차선이 사라졌습니다. 오른쪽으로 회전합니다.")
                                    left_speed = 47  # 왼쪽 바퀴 속도
                                    right_speed = 5  # 오른쪽 바퀴 속도
                                else:
                                    # 왼쪽 차선이 사라진 것으로 판단하고 왼쪽으로 회전
                                    print("왼쪽 차선이 사라졌습니다. 왼쪽으로 회전합니다.")
                                    left_speed = 5  # 왼쪽 바퀴 속도
                                    right_speed = 47  # 오른쪽 바퀴 속도
                            else:
                                # 이전 프레임의 정보가 없는 경우 기본 동작 설정
                                print("차선 정보가 부족하여 기본 동작을 수행합니다.")
                                left_speed = 30
                                right_speed = 30
                        else:
                            # 두 차선 모두 정상적으로 감지된 경우
                            mid_center_x = int((left_center_x + right_center_x) / 2)
                            mid_center_y = int((left_center_y + right_center_y) / 2)
                            cv2.circle(undist_frame, (mid_center_x, mid_center_y), 5, (255, 0, 0), -1)

                            # 이미지 중앙과의 편차 계산
                            image_center_x = undist_frame.shape[1] // 2
                            deviation = mid_center_x - image_center_x

                            # 모터 값 계산 및 전송
                            max_speed = 80  # 최대 속도
                            left_base_speed = 30  # 기본 속도****************
                            right_base_speed = 34
                            max_deviation = undist_frame.shape[1] // 2

                            normalized_deviation = deviation / max_deviation
                            normalized_deviation = np.clip(normalized_deviation, -1, 1)

                            left_speed = left_base_speed - (normalized_deviation * left_base_speed)
                            right_speed = right_base_speed + (normalized_deviation * right_base_speed)

                            left_speed = int(np.clip(left_speed, 0, max_speed))
                            right_speed = int(np.clip(right_speed, 0, max_speed))
                    elif left_center_x is not None and right_center_x is None:
                        # 오른쪽 차선이 사라진 경우
                        if prev_right_center_x is not None:
                            # 남아있는 왼쪽 차선의 위치가 이전 오른쪽 차선의 위치와 가까운지 확인
                            if abs(left_center_x - prev_right_center_x) < abs(left_center_x - prev_left_center_x):
                                # 왼쪽 차선이 사라지고 오른쪽 차선만 남은 것으로 판단
                                print("왼쪽 차선이 사라졌습니다. 왼쪽으로 회전합니다.")
                                left_speed = 5  # 왼쪽 바퀴 속도
                                right_speed = 47  # 오른쪽 바퀴 속도
                            else:
                                print("오른쪽 차선이 사라졌습니다. 오른쪽으로 회전합니다.")
                                left_speed = 47  # 왼쪽 바퀴 속도
                                right_speed = 5  # 오른쪽 바퀴 속도
                        else:
                            # 이전 오른쪽 차선 정보가 없을 때 기본적으로 오른쪽으로 회전
                            print("오른쪽 차선이 사라졌습니다. 오른쪽으로 회전합니다.")
                            left_speed = 47
                            right_speed = 5
                    elif left_center_x is None and right_center_x is not None:
                        # 왼쪽 차선이 사라진 경우
                        if prev_left_center_x is not None:
                            # 남아있는 오른쪽 차선의 위치가 이전 왼쪽 차선의 위치와 가까운지 확인
                            if abs(right_center_x - prev_left_center_x) < abs(right_center_x - prev_right_center_x):
                                # 오른쪽 차선이 사라지고 왼쪽 차선만 남은 것으로 판단
                                print("오른쪽 차선이 사라졌습니다. 오른쪽으로 회전합니다.")
                                left_speed = 47  # 왼쪽 바퀴 속도
                                right_speed = 5  # 오른쪽 바퀴 속도
                            else:
                                print("왼쪽 차선이 사라졌습니다. 왼쪽으로 회전합니다.")
                                left_speed = 5  # 왼쪽 바퀴 속도
                                right_speed = 47  # 오른쪽 바퀴 속도
                        else:
                            # 이전 왼쪽 차선 정보가 없을 때 기본적으로 왼쪽으로 회전
                            print("왼쪽 차선이 사라졌습니다. 왼쪽으로 회전합니다.")
                            left_speed = 5
                            right_speed = 47
                    else:
                        # 차선이 모두 감지되지 않은 경우, 속도를 감소하거나 정지
                        print("차선이 감지되지 않았습니다. 속도를 감소합니다.")
                        left_speed = 0
                        right_speed = 0

                    # 현재 프레임의 차선 위치를 저장하여 다음 프레임에서 사용
                    prev_left_center_x = left_center_x
                    prev_right_center_x = right_center_x

                    # 화면에 디버그용으로 표시
                    lane_overlay = np.zeros_like(undist_frame)
                    lane_overlay[preprocessImage == 255] = [0, 0, 255]
                    combined = cv2.addWeighted(undist_frame, 0.7, lane_overlay, 1.0, 0)
                    cv2.imshow("Lane Detection", combined)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
        start = 60 #using 60 as start byte instead of b'MC'
        motor_command = start.to_bytes(1, byteorder="big") + left_speed.to_bytes(1, byteorder="big") + right_speed.to_bytes(1, byteorder="big") + b"\n"

        # b'MC' 쓸거면
        # start = b'MC'
        # motor_command = start + left_speed.to_bytes(1, byteorder="big") + right_speed.to_bytes(1, byteorder="big") + b"\n" 

        try:
            rpi_conn.sendall(motor_command)
            print(f"send motor_command {motor_command}")
        except Exception as e:
            print(f"Error sending motor command: {e}")

        time.sleep(1)  # 2~5초에 한번 쏴주는것만으로도 충분함. 타임 슬립없으면 충돌남.


def detect_flower_info(pollination_conn):
    global robot_state
    try:
        while True:
            if robot_state == 0:
                continue
            header = b''
            while len(header) < 1:
                header += pollination_conn.recv(1 - len(header))

            # detect data
            if int.from_bytes(header, byteorder="big") == 22:
                size_data = b''
                print("get in header pollination")
                while len(size_data) < 2:
                    packet = pollination_conn.recv(2 - len(size_data))
                    if not packet:
                        break
                    size_data += packet
                data_size = int.from_bytes(size_data, byteorder="big")
                
                detect_data = b''
                while len(detect_data) < data_size:
                    packet = pollination_conn.recv(data_size - len(detect_data))
                    if not packet:
                        break
                    detect_data += packet

                pollination_conn.recv(1)  # \n 받기
                
                before_idx = 0
                for i in range(10, data_size, 10):
                    data = detect_data[before_idx : i]
                    before_idx = i
                    class_id = int.from_bytes(data[:2], byteorder="big")
                    x1 = int.from_bytes(data[2:4], byteorder="big")
                    y1 = int.from_bytes(data[4:6], byteorder="big")
                    x2 =int.from_bytes(data[6:8], byteorder="big")
                    y2 = int.from_bytes(data[8:], byteorder="big")
                    print(f"class : {class_id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

    except Exception as e:
        print(f"Error receiving pollination data: {e}")

def get_gui_cmd(gui_conn, rpi_conn):
    global robot_state
    try:
        while True:
            header = b''
            while len(header) < 2:
                header += gui_conn.recv(2 - len(header))
            print(f"header: {header}")

            # detect Robot Control command
            if header == b'RC':
                data = gui_conn.recv(2)
                state = int.from_bytes(data[:1], byteorder="big")
                print(f"get robot state: {state}")
                robot_state = state
                dat = header + data
                print(f"data {dat}")
                rpi_conn.sendall(header + data)

            # detect set schedule data
            elif header == b'SS':
                schedule_data = b''
                print(f"schedule : {schedule_data}")
                while len(schedule_data) < 5:
                    packet = gui_conn.recv(5 - len(schedule_data))
                    if not packet:
                        break
                    schedule_data += packet

                gui_conn.recv(1)  # \n 받기

                schedule_time_str = schedule_data.decode('utf-8')
                print(f"time scheduled at {schedule_time_str}")
                
                schedule_time = datetime.strptime(schedule_time_str, "%H:%M").time()
                print(f"schedule_time : {schedule_time}")
                while True:
                    current_datettime = datetime.now()
                    # current_time = today.strftime("%H:%M")
                    schedule_datetime = current_datettime.replace(hour=schedule_time.hour, minute=schedule_time.minute, second=0, microsecond=0)

                    time_difference = (schedule_datetime - current_datettime).total_seconds()
                    if time_difference <= 0:
                        schedule_datetime += timedelta(days=1)
                        sleep_duration = (schedule_datetime - current_datettime).total_seconds()
                        print("time difference <=0 :", time_difference)
                    if time_difference > 60: #60 seconds
                        sleep_duration = min(time_difference / 2, 600)  # Sleep for half the remaining time or a maximum of 10 minutes
                        print("time difference > 60", time_difference)
                    else:
                        print("else:", time_difference)
                        sleep_duration = time_difference  # If less than 60 seconds, sleep for the remaining time

                    print(f"Sleeping for {sleep_duration} seconds. Time until scheduled event: {time_difference} seconds")
                    time.sleep(sleep_duration)

                    current_datettime = datetime.now()
                    if current_datettime >= schedule_datetime:
                        # Send motor values to Raspberry Pi (robot)
                        print("Scheduled time reached! send motor value")
                        send_motor_value()
                        break


    except Exception as e:
        print(f"Error receiving robot state data: {e}")

def send_robot_state(gui_conn, rpi_conn):
    global robot_state
    while True:
        if robot_state == 1:
            time.sleep(20)
            try:
                cv2.destroyAllWindows()
                command = 0
                start = b'RS'
                send_data = start + command.to_bytes(1, byteorder="big") + b'\n'
                gui_conn.sendall(send_data)
                rpi_conn.sendall(send_data)
                print(f"send robot state: {command}")

                robot_state = 0

            except Exception as e:
                print(f"Error sending robot state data: {e}")


                
if __name__ == "__main__" :

    # 서버 설정
    server_ip = "192.168.0.134"
    gui_server_port = 1235
    rpi_server_port = 3141
    pollination_server_port = 8887
    obs_server_port = 4040

    gui_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    gui_server_socket.bind((server_ip, gui_server_port))
    gui_server_socket.listen(1)
    print(f"서버가 {server_ip} : {gui_server_port}에서 대기 중입니다...")

    rpi_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rpi_server_socket.bind((server_ip, rpi_server_port))
    rpi_server_socket.listen(1)
    print(f"서버가 {server_ip} : {rpi_server_port}에서 대기 중입니다...")

    obs_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    obs_server_socket.bind((server_ip, obs_server_port))
    obs_server_socket.listen(1)
    print(f"서버가 {server_ip} : {obs_server_port}에서 대기 중입니다...")

    pollination_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pollination_server_socket.bind((server_ip, pollination_server_port))
    pollination_server_socket.listen(1)
    print(f"서버가 {server_ip} : {pollination_server_port}에서 대기 중입니다...")

    gui_conn, addr = gui_server_socket.accept()
    print(f"클라이언트 {addr}와 연결되었습니다.")

    rpi_conn, addr = rpi_server_socket.accept()
    print(f"클라이언트 {addr}와 연결되었습니다.")

    obs_conn, addr = obs_server_socket.accept()
    print(f"클라이언트 {addr}와 연결되었습니다.")

    pollination_conn, addr = pollination_server_socket.accept()
    print(f"클라이언트 {addr}와 연결되었습니다.")

    rpi_client_thread = Thread(target=handle_rpi_client, args=(rpi_conn, obs_conn))
    motor_cmd_thread = Thread(target=send_motor_value, args=(rpi_conn,))
    pollination_thread = Thread(target=detect_flower_info, args=(pollination_conn,))
    send_gui_thread = Thread(target=send_robot_state, args=(gui_conn, rpi_conn))
    get_gui_thread = Thread(target=get_gui_cmd, args=(gui_conn, rpi_conn))

    rpi_client_thread.start()
    motor_cmd_thread.start()
    pollination_thread.start()
    send_gui_thread.start()
    get_gui_thread.start()
    
    rpi_client_thread.join()
    motor_cmd_thread.join()
    pollination_thread.join()
    send_gui_thread.join()
    get_gui_thread.join()

    rpi_conn.close()
    rpi_server_socket.close()
    pollination_conn.close()
    pollination_server_socket.close()
    gui_conn.close()
    gui_server_socket.close()
    cv2.destroyAllWindows()