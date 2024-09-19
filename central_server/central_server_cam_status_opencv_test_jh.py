import cv2, socket, struct, time, sys
import numpy as np

def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    else:
        raise ValueError("orient must be 'x' or 'y'")

    if np.max(abs_sobel) == 0:
        scaled_sobel = np.uint8(abs_sobel)
    else:
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

# central server ip, port
server_ip = "192.168.0.50"
rpi_server_port = 3141
obs_server_port = 4040

# 카메라 매트릭스과 왜곡 계수 (예시 값)
mtx = np.array([[1.15753008e+03, 0.00000000e+00, 6.75382833e+02],
                [0.00000000e+00, 1.15189955e+03, 3.86729350e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.26706898,  0.10305542, -0.00088013,  0.00080643, -0.19574027]])

# 소켓 생성 및 바인딩
rpi_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rpi_server_socket.bind((server_ip, rpi_server_port))
rpi_server_socket.listen(1)
obs_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
obs_server_socket.bind((server_ip, obs_server_port))
obs_server_socket.listen(1)
print(f"서버가 {server_ip} : {rpi_server_port}에서 대기 중입니다...")
print(f"서버가 {server_ip} : {obs_server_port}에서 대기 중입니다...")

# 클라이언트 연결 수립
rpi_conn, addr = rpi_server_socket.accept()
print(f"클라이언트 {addr}와 연결되었습니다.")
obs_conn, addr = obs_server_socket.accept()
print(f"클라이언트 {addr}와 연결되었습니다.")

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

            rpi_conn.recv(1)
            
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                # === 카메라 캘리브레이션 적용
                undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

                # === 차선 검출
                # color threshold
                hls = cv2.cvtColor(undist_frame, cv2.COLOR_BGR2HLS)
                l_channel = hls[:, :, 1]

                binary_output = np.zeros_like(l_channel)
                binary_output[(l_channel >= 0) & (l_channel <= 255)] = 1

                c_binary = binary_output
                # abs_sobel_thresh
                gradx = abs_sobel_thresh(undist_frame, orient='x', thresh_min=20, thresh_max=100)
                grady = abs_sobel_thresh(undist_frame, orient='y', thresh_min=20, thresh_max=100)

                combined_binary = np.zeros_like(c_binary)
                combined_binary[((gradx == 1) & (grady == 1)) & (c_binary == 1)] = 255

                # apply_roi
                height, width = combined_binary.shape[:2]
                mask = np.zeros_like(combined_binary)

                polygon = np.array([[
                    (0, int(height * 2/5)),
                    (width, int(height * 2/5)),
                    (width, height),
                    (0, height)
                ]], np.int32)

                cv2.fillPoly(mask, polygon, 255)
                roi_image = cv2.bitwise_and(combined_binary, mask)

                preprocessImage = roi_image

                # === 차선 인식 결과 계산
                # 이미지 중간 지점 계산
                midpoint = preprocessImage.shape[1] // 2

                # 이진 이미지에서 차선 픽셀 좌표 추출
                nonzero = preprocessImage.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])

                # 왼쪽 및 오른쪽 차선 필셀 식별
                left_lane_inds = (nonzerox < midpoint)
                right_lane_inds = (nonzerox >= midpoint)

                # 왼쪽 차선 픽셀 좌표
                leftx = nonzerox[left_lane_inds]
                lefty = nonzeroy[left_lane_inds]

                # 오른쪽 차선 픽셀 좌표
                rightx = nonzerox[right_lane_inds]
                righty = nonzeroy[right_lane_inds]

                # 왼쪽 차선의 무게중심 계산 및 표시
                if len(leftx) > 0 and len(lefty) > 0:
                    left_center_x = int(np.mean(leftx))
                    left_center_y = int(np.mean(lefty))
                    cv2.circle(undist_frame, (left_center_x, left_center_y), 5, (0, 255, 0), -1)  # 초록색 원
                else:
                    left_center_x = None
                    left_center_y = None

                # 오른쪽 차선의 무게중심 계산 및 표시
                if len(rightx) > 0 and len(righty) > 0:
                    right_center_x = int(np.mean(rightx))
                    right_center_y = int(np.mean(righty))
                    cv2.circle(undist_frame, (right_center_x, right_center_y), 5, (0, 255, 0), -1)  # 초록색 원
                else:
                    right_center_x = None
                    right_center_y = None

                # 두 차선 무게중심의 중앙점 계산 및 표시
                if left_center_x is not None and right_center_x is not None:
                    mid_center_x = int((left_center_x + right_center_x) / 2)
                    mid_center_y = int((left_center_y + right_center_y) / 2)
                    cv2.circle(undist_frame, (mid_center_x, mid_center_y), 5, (255, 0, 0), -1)  # 파란색 원
                else:
                    mid_center_x = None
                    mid_center_y = None

                # === 거리 계산 및 표시 ===
                meters_per_pixel_top = 0.05  # 예시 값 (상단 픽셀당 미터)
                meters_per_pixel_bottom = 0.05  # 예시 값 (하단 픽셀당 미터)

                if left_center_x is not None and mid_center_x is not None and left_center_y is not None:
                    image_height = undist_frame.shape[0]
                    y_ratio = left_center_y / image_height

                    meters_per_pixel = meters_per_pixel_top + (meters_per_pixel_bottom - meters_per_pixel_top) * y_ratio

                    pixel_distance = abs(mid_center_x - left_center_x)

                    real_distance_meters = pixel_distance * meters_per_pixel

                    distance_text = f"Distance from Left Lane: {real_distance_meters:.2f} meters"

                    cv2.putText(undist_frame, distance_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(undist_frame, "Lane detection failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)

                # === 객체 검출 로직 (예: 특정 색상 객체 감지) ===
                hsv = cv2.cvtColor(undist_frame, cv2.COLOR_BGR2HSV)
                # 예시: 빨간색 범위 (HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 70, 50])
                upper_red2 = np.array([180, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                object_detected = 0  # 기본값은 객체 미검출
                if len(contours) > 0:
                    # 객체가 차선 내부에 있는지 확인
                    # 간단히 중앙 영역에 객체가 있는지 확인
                    central_region = preprocessImage[int(preprocessImage.shape[0]*0.6):, :]
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 좌표 변환: y는 이미지 상단부터 시작
                        if y > int(preprocessImage.shape[0]*0.6):
                            object_detected = 1
                            cv2.rectangle(undist_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # === 모터 값 전송
                # determine_motor_value
                height, width = preprocessImage.shape
                midpoint = width // 2
                left_region = preprocessImage[int(height*0.6):, :midpoint]
                right_region = preprocessImage[int(height*0.6):, midpoint:]

                left_count = np.sum(left_region) / 255
                right_count = np.sum(right_region) / 255

                # 비교를 통해 차선의 위치를 판단
                direction = b''
                if left_count > right_count + 50:
                    motor_value = 1500 # 우회전
                    direction = b'R'
                elif right_count > left_count + 50:
                    motor_value = 500 # 좌회전
                    direction = b'L'
                else:
                    motor_value = 1000 # 직진
                    direction = b'M'
                
                motor_command = b'M' + direction + motor_value.to_bytes(2, byteorder="big") + b'\n'
                rpi_conn.sendall(motor_command)
                print(f"send motor_command {direction}, {motor_value}")

                # === Obstacle 서버로 영상 전송
                encoded_cal_frame = cv2.imencode('.jpg', undist_frame)[1].tobytes()
                cal_frame_size = len(frame_data)
                head = b'SF'
                obs_conn.sendall(head + struct.pack(">L", cal_frame_size) + encoded_cal_frame + b'\n')

                # === 화면에 표시 (디버그용)
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
            
            rpi_conn.recv(1)

            status = int.from_bytes(status_data, byteorder="big")
            if status == 1:
                print("status:", status)
            else:
                print("status", status)

except Exception as e:
    print(f"Error receiving or displaying frame: {e}")

finally:
    rpi_conn.close()
    rpi_server_socket.close()
    cv2.destroyAllWindows()
