import cv2
from cv2 import aruco
import numpy as np
import socket
import struct
import time
from threading import Thread

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
    upper_white = np.array([60, 10, 255])
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
global state, state_start_time

# ARUCO 마커의 실제 크기 (미터 단위)
MARKER_SIZE = 0.0195  # 마커의 실제 크기 (미터 단위)

# 이미지에 텍스트를 그리는 함수
def draw_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
              scale=0.6, color=(0, 0, 255), thickness=1):
    cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)

def handle_client(rpi_conn):
    global left_speed, right_speed
    global state, state_start_time
    state = 'lane_following'
    state_start_time = None

    # 이전 프레임의 차선 무게중심 좌표 저장 변수 초기화
    prev_left_center_x = None
    prev_right_center_x = None
    prev_left_center_y = None
    prev_right_center_y = None

    try:
        while True:
            header = b''
            while len(header) < 2:
                packet = rpi_conn.recv(2 - len(header))
                if not packet:
                    break
                header += packet

            # 프레임 데이터
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

                rpi_conn.recv(1)  # '\n' 받기

                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # 카메라 캘리브레이션 적용
                    undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

                    # ARUCO 마커 딕셔너리 생성 및 파라미터 설정
                    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
                    param_markers = aruco.DetectorParameters()

                    # ARUCO 마커 그레이스케일로 변환
                    gray_frame = cv2.cvtColor(undist_frame, cv2.COLOR_BGR2GRAY)
                    marker_corners, marker_IDs, reject = aruco.detectMarkers(
                        gray_frame, marker_dict, parameters=param_markers
                    )

                    # 아르코 마커 위치 초기화
                    cam_x, cam_y, cam_z = None, None, None

                    # ARUCO 마커가 검출되면
                    if marker_corners and marker_IDs is not None:
                        # 포즈 추정
                        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                            marker_corners, MARKER_SIZE, mtx, dist
                        )
                        for i, ids in enumerate(marker_IDs):
                            # 마커 경계 그리기
                            aruco.drawDetectedMarkers(undist_frame, marker_corners)

                            # 마커의 중심을 원점으로 설정하고 차량의 위치 계산
                            R_ct = np.matrix(cv2.Rodrigues(rvecs[i])[0])
                            R_tc = R_ct.T
                            camera_position = -R_tc * np.matrix(tvecs[i][0]).T
                            camera_position = camera_position * 100  # cm로 변환
                            cam_x, cam_y, cam_z = camera_position.A1

                            # 표시할 텍스트 준비
                            text = f"x={cam_x:.1f}cm y={cam_y:.1f}cm z={cam_z:.1f}cm"

                            # 마커의 중심 찾기
                            corners = marker_corners[i].reshape(4, 2)
                            marker_center = np.mean(corners, axis=0).astype(int)
                            cv2.circle(undist_frame, tuple(marker_center), 5, (0, 0, 255), -1)

                            # 이미지에 텍스트 그리기
                            draw_text(undist_frame, text, (marker_center[0] + 10, marker_center[1] + 10))

                            # x, y, z 값 출력
                            print(f"x={cam_x:.1f}cm, y={cam_y:.1f}cm, z={cam_z:.1f}cm")

                            # 좌표축 직접 그리기
                            axis_length = 0.01  # 1cm
                            axis_points = np.float32([
                                [0, 0, 0],  # 원점
                                [axis_length, 0, 0],  # x축
                                [0, axis_length, 0],  # y축
                                [0, 0, axis_length]   # z축
                            ]).reshape(-1, 3)

                            # 이미지 평면으로 투영
                            imgpts, jac = cv2.projectPoints(axis_points, rvecs[i], tvecs[i], mtx, dist)
                            imgpts = np.int32(imgpts).reshape(-1, 2)

                            # 원점과 각 축 끝점을 연결하여 그리기
                            origin = tuple(imgpts[0])
                            cv2.line(undist_frame, origin, tuple(imgpts[1]), (0, 0, 255), 2)  # x축 (빨강)
                            cv2.line(undist_frame, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # y축 (초록)
                            cv2.line(undist_frame, origin, tuple(imgpts[3]), (255, 0, 0), 2)  # z축 (파랑)
                        
                        if ids == 5 and cam_z < 10:
                            state = 't-junction'

                        elif ids == 6 and cam_z < 10:
                            state = 'crossroad'

                        else:
                            state = 'lane_following'

                    else:
                        print("아르코 마커가 검출되지 않았습니다.")

                    # 여기서 상태 관리를 시작합니다.
                    # x_min, x_max, y_min, y_max, z_min, z_max를 설정합니다.
                    x_min, x_max = -1, 0.9   # cm
                    y_min, y_max = 11.0, 30  # cm
                    z_min, z_max = 0.0, 30.0  # cm

                    if state == 'lane_following':
                        # (1) 차선 검출: x, y 경계값
                        gradx = abs_sobel_thresh(undist_frame, orient='x', thresh_min=50, thresh_max=255)
                        grady = abs_sobel_thresh(undist_frame, orient='y', thresh_min=50, thresh_max=255)

                        # (2) 차선 검출: 색상 임계값
                        c_binary = color_threshold(undist_frame)

                        # 최종 이진 이미지 생성
                        preprocessImage = np.zeros_like(undist_frame[:, :, 0])
                        preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

                        # ROI 적용
                        preprocessImage = apply_roi(preprocessImage)

                        # 프레임 너비 계산
                        frame_width = preprocessImage.shape[1]

                        # 왼쪽 ROI 설정 (프레임의 왼쪽 1/3)
                        left_roi_x_end = int(frame_width * 1/3)
                        cv2.line(undist_frame, (left_roi_x_end, 0), (left_roi_x_end, undist_frame.shape[0]), (255, 255, 0), 2)

                        # 오른쪽 ROI 설정 (프레임의 오른쪽 1/3)
                        right_roi_x_start = int(frame_width * 2/3)
                        cv2.line(undist_frame, (right_roi_x_start, 0), (right_roi_x_start, undist_frame.shape[0]), (255, 255, 0), 2)

                        # 왼쪽 ROI에서 무게중심 계산
                        nonzero = preprocessImage.nonzero()
                        nonzeroy = np.array(nonzero[0])
                        nonzerox = np.array(nonzero[1])

                        left_lane_inds = (nonzerox >= 0) & (nonzerox < left_roi_x_end)
                        leftx = nonzerox[left_lane_inds]
                        lefty = nonzeroy[left_lane_inds]

                        # 오른쪽 ROI에서 무게중심 계산
                        right_lane_inds = (nonzerox >= right_roi_x_start) & (nonzerox < frame_width)
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

                        # 차선 기반 모터 제어 코드

                        # 두 차선 모두 감지된 경우
                        if left_center_x is not None and right_center_x is not None:
                            lane_distance = abs(right_center_x - left_center_x)
                            print(f"Lane Width: {lane_distance}px")

                            if lane_distance < 400:
                                # 하나의 차선이 사라진 것으로 간주
                                print("차선 간격이 좁습니다. 하나의 차선이 사라졌다고 판단합니다.")
                                # 어느 차선이 사라졌는지 판단
                                if left_center_x < frame_width / 2:
                                    # 왼쪽 차선만 감지됨, 오른쪽 차선이 사라짐
                                    print("오른쪽 차선이 사라졌습니다. 오른쪽으로 회전합니다.")
                                    left_speed = 5
                                    right_speed = 43
                                else:
                                    # 오른쪽 차선만 감지됨, 왼쪽 차선이 사라짐
                                    print("왼쪽 차선이 사라졌습니다. 왼쪽으로 회전합니다.")
                                    left_speed = 43
                                    right_speed = 5
                            else:
                                # 정상적인 차선 추종
                                mid_center_x = int((left_center_x + right_center_x) / 2)
                                mid_center_y = int((left_center_y + right_center_y) / 2)
                                cv2.circle(undist_frame, (mid_center_x, mid_center_y), 5, (255, 0, 0), -1)

                                # 왼쪽 경계로부터의 거리 계산
                                distance_from_left = mid_center_x
                                max_distance = frame_width  # 최대 거리

                                # 거리를 0부터 1 사이로 정규화
                                normalized_distance = distance_from_left / max_distance
                                normalized_distance = np.clip(normalized_distance, 0, 1)

                                # 왼쪽에서 멀어질수록 회전 강도를 높임
                                max_turn_speed = 100  # 최대 회전 속도 차이

                                # 중앙부에 ROI 영역 설정 (디버깅을 위해)
                                roi_width = 30  # ROI의 너비
                                roi_left_boundary = (frame_width - roi_width) // 2
                                roi_right_boundary = roi_left_boundary + roi_width

                                # ROI 영역 시각화
                                cv2.line(undist_frame, (roi_left_boundary, 0), (roi_left_boundary, undist_frame.shape[0]), (0, 255, 255), 2)
                                cv2.line(undist_frame, (roi_right_boundary, 0), (roi_right_boundary, undist_frame.shape[0]), (0, 255, 255), 2)

                                if roi_left_boundary <= mid_center_x <= roi_right_boundary:
                                    # ROI 내에 있으면 직진 주행
                                    left_speed = 30  # 기본 속도
                                    right_speed = 34  # 기본 속도
                                    print("중앙선 내에 있습니다. 직진 주행합니다.")
                                    # 거리 비율 표시
                                    cv2.putText(undist_frame, f"Distance Ratio: {normalized_distance:.2f}", (50, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                else:
                                    # 왼쪽에서 멀어진 정도에 따라 회전 강도 조절
                                    if mid_center_x < roi_left_boundary:
                                        # 왼쪽으로 치우쳤으므로 오른쪽으로 회전해야 함
                                        turn_intensity = (roi_left_boundary - mid_center_x) / roi_left_boundary
                                        left_speed = 30 - (turn_intensity * max_turn_speed + 5)
                                        right_speed = 34 + (turn_intensity * max_turn_speed + 5)
                                        print(f"왼쪽으로 치우쳤습니다. 오른쪽으로 회전합니다. 거리 비율: {turn_intensity:.2f} | ML: {left_speed}, MR: {right_speed}")
                                        # 거리 비율 표시
                                        cv2.putText(undist_frame, f"Distance Ratio: {turn_intensity:.2f}", (50, 100),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        # 화살표 그리기
                                        start_point = (roi_left_boundary, mid_center_y)
                                        end_point = (mid_center_x, mid_center_y)
                                        cv2.arrowedLine(undist_frame, start_point, end_point, (0, 0, 255), 3, tipLength=0.5)
                                    else:
                                        # 오른쪽으로 치우쳤으므로 왼쪽으로 회전해야 함
                                        turn_intensity = (mid_center_x - roi_right_boundary) / (frame_width - roi_right_boundary)
                                        left_speed = 30 + (turn_intensity * max_turn_speed + 5)
                                        right_speed = 34 - (turn_intensity * max_turn_speed + 5)
                                        print(f"오른쪽으로 치우쳤습니다. 왼쪽으로 회전합니다. 거리 비율: {turn_intensity:.2f} | ML: {left_speed}, MR: {right_speed}")
                                        # 거리 비율 표시
                                        cv2.putText(undist_frame, f"Distance Ratio: {turn_intensity:.2f}", (50, 100),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        # 화살표 그리기
                                        start_point = (roi_right_boundary, mid_center_y)
                                        end_point = (mid_center_x, mid_center_y)
                                        cv2.arrowedLine(undist_frame, start_point, end_point, (0, 0, 255), 3, tipLength=0.5)

                                # 속도 값이 유효한 범위 내에 있도록 제한
                                max_speed = 80  # 최대 속도
                                left_speed = int(np.clip(left_speed, 0, max_speed))
                                right_speed = int(np.clip(right_speed, 0, max_speed))

                        # 왼쪽 차선만 감지된 경우
                        elif left_center_x is not None and right_center_x is None:
                            print("오른쪽 차선이 감지되지 않았습니다. 오른쪽으로 회전합니다.")
                            left_speed = 43
                            right_speed = 5

                        # 오른쪽 차선만 감지된 경우
                        elif left_center_x is None and right_center_x is not None:
                            print("왼쪽 차선이 감지되지 않았습니다. 왼쪽으로 회전합니다.")
                            left_speed = 5
                            right_speed = 43

                        else:
                            # 차선이 모두 감지되지 않은 경우, 정지
                            print("차선이 감지되지 않았습니다. 정지합니다.")
                            if ids is not None:
                                if ids[0] == 6:
                                    remain = 0
                                    left_speed = 30 - remain
                                    right_speed = 34 + remain
                                if ids[0] == 5:
                                    remain = 5
                                    left_speed = 30 - remain
                                    right_speed = 34 + remain
                            left_speed = 0
                            right_speed = 0

                    # ARUCO 마커가 지정된 범위 내에 있는지 확인
                    if state == 'crossroad':
                        # # 상태를 'aruco_stopped'로 전환
                        # print("ARUCO 마커가 범위 내에 있습니다. 3초간 정지합니다.")
                        # left_speed = 0
                        # right_speed = 0
                        # state = 'aruco_stopped'
                        # state_start_time = time.time()
                        # elif state == 'aruco_stopped':
                        #     # 차량을 3초 동안 정지
                        #     left_speed = 0
                        #     right_speed = 0
                        #     elapsed = time.time() - state_start_time
                        #     if elapsed >= 3.0:
                        #         # 3초 후 전진 시작
                        #         left_speed = 26
                        #         right_speed = 30
                        #         state = 'moving_forward'
                        #         state_start_time = time.time()
                        #         print("3초간 정지 후 3초간 전진을 시작합니다.")

                        # elif state == 'moving_forward':
                        #     # 차량을 3초 동안 전진
                        #     left_speed = 26
                        #     right_speed = 30
                        #     elapsed = time.time() - state_start_time
                        #     if elapsed >= 3.0:
                        #         # 3초 후 차선 인식 모드로 전환
                        #         state = 'lane_following'
                        #         print("3초간 전진 후 차선 인식 모드로 전환합니다.")
                        pass

                    elif state == 't-junction':
                        # T자로 만나는 지점에서는 왼쪽으로 회전
                        left_speed = 20
                        right_speed = 36
                        elapsed = time.time() - state_start_time
                        if elapsed >= 2.0:
                            # 2초 후 전진 시작
                            left_speed = 26
                            right_speed = 30
                            state = 'lane_following'
                            state_start_time = time.time()
                            print("2초간 좌회전 후 차선 인식 모드로 전환합니다.")

                    else:
                        # 기타 상태에서는 아무것도 하지 않음
                        pass

                    # 화면에 디버그용으로 표시
                    if state == 'lane_following':
                        # 차선 인식 모드일 때만 차선 시각화를 표시
                        lane_overlay = np.zeros_like(undist_frame)
                        lane_overlay[preprocessImage == 255] = [0, 0, 255]
                        combined = cv2.addWeighted(undist_frame, 0.7, lane_overlay, 1.0, 0)
                        cv2.imshow("Lane Detection", combined)
                    else:
                        # 다른 상태에서는 원본 프레임만 표시
                        cv2.imshow("Lane Detection", undist_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # 상태 데이터
                elif header == b'CS':
                    status_data = b''
                    while len(status_data) < 1:
                        packet = rpi_conn.recv(1 - len(status_data))
                        if not packet:
                            break
                        status_data += packet

                    rpi_conn.recv(1)  # '\n' 받기

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
        except Exception as e:
            print(f"Error sending motor command: {e}")

        time.sleep(1)  # 없으면 충돌남

if __name__ == "__main__" :

    # 서버 설정
    server_ip = "192.168.0.147"
    rpi_server_port = 3140

    rpi_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rpi_server_socket.bind((server_ip, rpi_server_port))
    rpi_server_socket.listen(1)
    print(f"서버가 {server_ip} : {rpi_server_port}에서 대기 중입니다...")

    rpi_conn, addr = rpi_server_socket.accept()
    print(f"클라이언트 {addr}와 연결되었습니다.")

    client_thread = Thread(target=handle_client, args=(rpi_conn,))
    client_thread.start()

    motor_cmd_thread = Thread(target=send_motor_value, args=(rpi_conn,))
    motor_cmd_thread.start()

    client_thread.join()
    motor_cmd_thread.join()

    rpi_conn.close()
    rpi_server_socket.close()
    cv2.destroyAllWindows()
