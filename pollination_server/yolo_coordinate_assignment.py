import cv2 as cv
import socket
from cv2 import aruco
import numpy as np
import os
import time
from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# --------------------------------- Constants -------------------------------- #

CENTRAL_SERVER_IP = "192.168.1.11"
CENTRAL_SERVER_PORT = 8888

POLLINATION_SERVER_IP = "192.168.1.17"
POLLINATION_SERVER_PORT = 8888

# -------------- Server Connection & Camera Utility Functions ---------------- #

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

def get_external_camera():
    video_devices = []
    for device in os.listdir('/dev/'):
        if device.startswith('video'):
            video_devices.append(device)
    external_cam = 0  # Default to webcam
    for device in video_devices:
        device_id = int(device.replace('video', ''))
        cap_test = cv.VideoCapture(device_id)
        if cap_test.isOpened() and device_id != 0:
            print(f"Device /dev/{device} (ID: {device_id}) is available.")
            external_cam = device_id
            cap_test.release()
            break
    return external_cam

# ----------------------- Flower Monitoring Functions ------------------------ #

def get_object_centers(results):
    """
    YOLO 결과에서 객체의 중심 좌표를 추출합니다.
    """
    centers = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            class_num = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id not in centers:
                # centers dict에 track_id key를 추가, value는 빈 리스트로 초기화
                centers[track_id] = []
            # dict에 (track_id, (center_x, center_y), class_num) 튜플 추가
            centers[track_id].append((track_id, (center_x, center_y), class_num))
    return centers

def estimate_camera_pose(marker_corners, marker_ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs):
    """
    함수의 기능 : 검출된 ArUco 마커의 지정된 위치를 이용해 카메라 자세를 추정
    (1) 카메라 내부 파라미터(K)는 캘리브레이션 데이터에서 가져오기
    (2) solvePnP를 사용하여 3D 공간상의 마커위치를 2D로 투영 후 카메라의 회전 및 위치를 계산
    (3) 최적화 알고리즘은 SOLVEPNP_ITERATIVE를 사용
    반환 값 : 회전 벡터(rvec)와 이동 벡터(tvec)를 반환
    """
    object_points = []  # 월드 좌표계 3D 포인트, ArUco marker의 중심을 기준으로 생성된 좌표를 사용할 예정
    image_points  = []  # 이미지 좌표계 2D 포인트
    half_size = marker_size_cm / 2.0
    # 각 ArUco의 설정된 3D 포인트 및 이미지 상의 2D 포인트를 설정
    for i, marker_id in enumerate(marker_ids.flatten()):
        if marker_id in marker_positions_cm:
            center_pos = marker_positions_cm[marker_id]
            # SOLVEPNP_ITERATIVE를 사용하기 위해 4개의 코너 좌표를 리스트에 저장
            obj_pts = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
            obj_pts += center_pos.reshape(1, 3)
            object_points.append(obj_pts)
            img_pts = marker_corners[i][0].reshape(4, 2)
            image_points.append(img_pts)
        else:
            print(f"Marker ID {marker_id} not in known marker positions.")
    
    if len(object_points) == 0:
        print("No known markers detected for pose estimation.")
        return None, None
    # 3D 포인트와 2D 포인트를 numpy 배열로 변환
    object_points = np.concatenate(object_points, axis=0).astype(np.float32)
    image_points = np.concatenate(image_points, axis=0).astype(np.float32)
    # solvePnP를 사용하여 카메라 자세 추정
    success, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    
    if not success:
        print("Camera pose estimation failed.")
        return None, None
    
    return rvec, tvec

# def measure_marker_distances(marker_corners, marker_ids, marker_size, camera_matrix, dist_coeffs):
#     """
#     각 ArUco 마커까지의 거리를 측정합니다.
#     """
#     distances = {}
#     scale_factor = 3.0 / marker_size  
#     for i in range(len(marker_ids)):
#         corners = marker_corners[i]
#         marker_id = marker_ids[i][0]
#         obj_points = np.array([
#             [-marker_size/2, marker_size/2, 0],
#             [marker_size/2, marker_size/2, 0],
#             [marker_size/2, -marker_size/2, 0],
#             [-marker_size/2, -marker_size/2, 0]
#         ], dtype=np.float32)
#         img_points = corners.reshape(-1, 2)
#         success, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
#         if success:
#             distance = np.linalg.norm(tvec) * scale_factor
#             distances[marker_id] = {'distance': distance, 'rvec': rvec, 'tvec': tvec}
#         else:
#             print(f"Pose estimation failed for marker ID {marker_id}")
#     return distances

def measure_marker_distances(marker_corners, marker_ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs):
    """
    각 ArUco 마커까지의 거리를 측정하고, 스케일 팩터를 계산합니다.
    """
    distances = {}
    scale_factors = []
    for i in range(len(marker_ids)):
        corners = marker_corners[i]
        marker_id = marker_ids[i][0]
        obj_points = np.array([
            [-marker_size_cm/2, marker_size_cm/2, 0],
            [marker_size_cm/2, marker_size_cm/2, 0],
            [marker_size_cm/2, -marker_size_cm/2, 0],
            [-marker_size_cm/2, -marker_size_cm/2, 0]
        ], dtype=np.float32)
        img_points = corners.reshape(-1, 2)
        success, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if success:
            tvec = tvec.flatten()
            estimated_distance = np.linalg.norm(tvec)
            if marker_id in marker_positions_cm:
                actual_position = marker_positions_cm[marker_id]
                actual_distance = np.linalg.norm(actual_position)
                scale_factor = actual_distance / estimated_distance
                scale_factors.append(scale_factor)
                print(f"Marker ID {marker_id}: Scale Factor = {scale_factor}")
                # Adjust tvec using the scale factor
                tvec *= scale_factor
                estimated_distance = np.linalg.norm(tvec)
            else:
                print(f"Marker ID {marker_id} position not known. Skipping scale factor computation.")
            distances[marker_id] = {'distance': estimated_distance, 'rvec': rvec, 'tvec': tvec}
        else:
            print(f"Pose estimation failed for marker ID {marker_id}")
    # Compute average scale factor
    if scale_factors:
        average_scale = np.mean(scale_factors)
        print(f"Average Scale Factor: {average_scale}")
    else:
        average_scale = None
        print("No scale factors computed. Check marker_positions_cm.")
    return distances, average_scale

def calibrate_3d_points(points_3d_marker, marker_positions_cm):
    """
    여러 마커를 사용하여 평균 스케일 팩터를 계산하고 3D 포인트를 보정합니다.
    """
    scale_factors = []
    for marker_id in points_3d_marker:
        if marker_id in marker_positions_cm:
            triangulated_point = points_3d_marker[marker_id]['point']
            actual_position = marker_positions_cm[marker_id]
            triangulated_distance = np.linalg.norm(triangulated_point)
            actual_distance = np.linalg.norm(actual_position)
            if triangulated_distance == 0:
                print(f"Triangulated distance for marker ID {marker_id} is zero. Skipping scale factor.")
                continue
            scale_factor = actual_distance / triangulated_distance
            scale_factors.append(scale_factor)
            print(f"Marker ID {marker_id}: Scale Factor = {scale_factor}")

    if scale_factors:
        average_scale = np.mean(scale_factors)
        print(f"Average Scale Factor: {average_scale}")
        # for obj_id in points_3d_marker:
            # points_3d_marker[obj_id] *= average_scale
        for obj_id in points_3d_marker:
            points_3d_marker[obj_id]['point'] = np.array(points_3d_marker[obj_id]['point']) * average_scale  # numpy array로 변환 후 곱셈

    return points_3d_marker

def triangulate_3d_points(captures, camera_matrix, dist_coeffs):
    points_3d_marker = {}
    for i in range(len(captures) - 1):
        for j in range(i + 1, len(captures)):
            capture1 = captures[i]
            capture2 = captures[j]

            R1, _ = cv.Rodrigues(capture1['rvec'])
            t1 = capture1['tvec'].reshape(3, 1)
            R2, _ = cv.Rodrigues(capture2['rvec'])
            t2 = capture2['tvec'].reshape(3, 1)
            P1 = camera_matrix @ np.hstack((R1, t1))
            P2 = camera_matrix @ np.hstack((R2, t2))

            centers1 = capture1['object_centers']
            centers2 = capture2['object_centers']
            common_ids = set(centers1.keys()).intersection(centers2.keys())

            if not common_ids:
                print(f"No common objects between capture {i} and capture {j}.")
                continue

            for obj_id in common_ids:
                pt1 = np.array(centers1[obj_id][0][1], dtype=np.float32).reshape(2, 1)
                pt2 = np.array(centers2[obj_id][0][1], dtype=np.float32).reshape(2, 1)
                class_id = centers1[obj_id][0][2]  # Assuming class_id is consistent across captures
                print(f"Triangulating for ID {obj_id}: Capture {i} (x={pt1[0][0]}, y={pt1[1][0]}), Capture {j} (x={pt2[0][0]}, y={pt2[1][0]})")
                point_4d_hom = cv.triangulatePoints(P1, P2, pt1, pt2)
                point_3d = cv.convertPointsFromHomogeneous(point_4d_hom.T)[0][0]  # (x, y, z)

                if obj_id not in points_3d_marker:
                    points_3d_marker[obj_id] = {'point': point_3d, 'class_id': class_id}
                else:
                    existing_point = points_3d_marker[obj_id]['point']
                    avg_point = np.mean([existing_point, point_3d], axis=0)
                    points_3d_marker[obj_id]['point'] = avg_point
                    # Ensure class IDs are consistent
                    if points_3d_marker[obj_id]['class_id'] != class_id:
                        print(f"Warning: Class ID mismatch for object ID {obj_id}")
    return points_3d_marker

def visualize_3d_points(points_3d_marker, save_dir, capture_time):
    """
    3D 포인트를 시각화하고 파일로 저장합니다.
    """
    if not points_3d_marker:
        print("No 3D points to visualize.")
        return

    points_array = np.array(list(points_3d_marker.values()))
    clustering = DBSCAN(eps=1.0, min_samples=2).fit(points_array)
    labels = clustering.labels_
    num_flowers = len(set(labels))
    print(f"\n클러스터링을 통해 추정된 꽃의 개수는 {num_flowers}개 입니다.")

    # 3D 포인트 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, (obj_id, point) in enumerate(points_3d_marker.items()):
        cluster_id = labels[idx]
        ax.scatter(point[0], point[1], point[2], label=f'Cluster {cluster_id}')
        ax.text(point[0], point[1], point[2], f'ID:{obj_id}')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.legend()
    plot_filename = os.path.join(save_dir, f"triangulated_3d_points_{capture_time}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"3D plot saved as {plot_filename}")

    return num_flowers

def count_flowers_by_distance(points_3d_marker, distance_threshold=5.0, class_num=None):
    """
    3D 포인트 간의 거리를 계산하여 서로 다른 꽃의 개수를 셉니다.
    포인트 간의 거리가 distance_threshold(cm) 이상이면 별개의 꽃으로 간주합니다.
    """
    unique_points = {}
    num_flowers = 0

    for obj_id, data in points_3d_marker.items():
        point = data['point']
        class_id = data['class_id']
        if class_num is not None and class_id != class_num:
            continue  # Skip points not matching the class_num

        duplicate_found = False

        # 기존에 저장된 unique_points와 거리를 계산하여 중복 여부 확인
        for up_id, up_data in unique_points.items():
            upoint = up_data['point']
            distance = np.linalg.norm(point - upoint)
            if distance < distance_threshold:
                duplicate_found = True
                break
        
        # 중복되지 않은 포인트는 새로운 꽃으로 간주
        if not duplicate_found:
            unique_points[obj_id] = point
            num_flowers += 1  # 새로운 꽃 추가

    return num_flowers, unique_points

def remove_duplicates(points_3d_list, distance_threshold=5.0):
    """
    중복되는 3D 포인트를 제거합니다. 포인트 간의 거리가 distance_threshold(cm) 미만이면 중복으로 간주합니다.
    """
    unique_points = []
    for point in points_3d_list:
        duplicate_found = False
        for upoint in unique_points:
            distance = np.linalg.norm(point - upoint)
            if distance < distance_threshold:
                duplicate_found = True
                break
        if not duplicate_found:
            unique_points.append(point)
    return unique_points

# ----------------------------- Main Function ----------------------------- #

def main():
    # central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    # if not central_sock:
    #     return
    # central_sock.settimeout(None)

    # last_sent_time = 0
    # detection_status = None
    # flag = False

    try:
        # 1. Parse Command-Line Arguments
        parser = argparse.ArgumentParser(description='YOLO Object Tracking with 3D Triangulation using Epipolar Geometry')
        parser.add_argument('--model', type=str, default='/home/ask/OneDrive/Document/dev_ws_DL/count_flower/best0x98.pt',
                            help='Path to the YOLOv8 model file')
        parser.add_argument('--calib_data', type=str, default='/home/ask/OneDrive/Document/dev_ws_DL/count_flower/logi_calibration.npz',
                            help='Path to the camera calibration data (.npz file)')
        parser.add_argument('--save_dir', type=str, default='captured_frames/home/ask/OneDrive/Document/dev_ws_DL/count_flower/5.YOLO_coordinate/save_dir',
                            help='Directory to save captured frames and data')
        parser.add_argument('--marker_size_cm', type=float, default=5.2,  # in centimeters
                            help='Size of the ArUco marker (in centimeters)')
        args = parser.parse_args()

        # 2. Load Camera Calibration Data
        if not os.path.exists(args.calib_data):
            print(f"Calibration data file not found at {args.calib_data}")
            return
        calib_data = np.load(args.calib_data)
        required_keys = ["camera_matrix", "dist_coeffs"]
        if not all(key in calib_data for key in required_keys):
            print(f"Calibration data must contain the following keys: {required_keys}")
            return
        print("Calibration data loaded successfully.")
        camera_matrix = calib_data["camera_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]

        # 3. Initialize YOLOv8 Model
        model = YOLO(args.model)

        # 4. Initialize Video Capture
        external_cam = get_external_camera()
        cap = cv.VideoCapture(external_cam)
        if not cap.isOpened():
            return print("카메라를 열 수 없습니다. 카메라 연결을 확인하세요.")

        # 5. Prepare Save Directory
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"저장 디렉토리를 생성했습니다: {save_dir}")

        # 6. Set Up ArUco Marker Detector
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        parameters = aruco.DetectorParameters()

        # 7. Known Marker Positions (in cm)
        marker_positions_cm = {
            1: np.array([10.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([20.0, 0.0, 0.0], dtype=np.float32),
            3: np.array([30.0, 0.0, 0.0], dtype=np.float32),
            4: np.array([50.0, 0.0, 50.0], dtype=np.float32),
            5: np.array([40.0, 0.0, 0.0], dtype=np.float32),
            6: np.array([50.0, 0.0, 0.0], dtype=np.float32),
            7: np.array([60.0, 0.0, 0.0], dtype=np.float32),
            8: np.array([70.0, 0.0, 0.0], dtype=np.float32),
            9: np.array([80.0, 0.0, 0.0], dtype=np.float32),
            10: np.array([90.0, 0.0, 0.0], dtype=np.float32),
            11: np.array([60.0, 10.0, 50.0], dtype=np.float32),
            12: np.array([70.0, 20.0, 50.0], dtype=np.float32),
            13: np.array([0.0, 0.0, 0.0], dtype=np.float32),
            14: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        marker_size_cm = args.marker_size_cm

        # 8. Initialize Variables for Captures
        saved_frames = []  # List to store captured frames
        results_list = []  # List to store results for each frame

        # 9. Main Processing Loop
        print("프레임을 저장하려면 's'버튼을 누르세요, 프로세스를 실행하려면 'd'버튼을 누르세요, 종료하려면 'q'버튼을 누르세요.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # Undistort and rotate the frame
            frame_undist = cv.undistort(frame, camera_matrix, dist_coeffs)
            rotated_frame_undist = cv.rotate(frame_undist, cv.ROTATE_90_COUNTERCLOCKWISE)

            # Convert to grayscale for ArUco detection
            gray = cv.cvtColor(rotated_frame_undist, cv.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            # if ids is not None:
            #     # print(f"Detected corners: {corners}, IDs: {ids}")

            # ----------------- Server Connection & Data Transmission ----------------- #
            
            # 현재 시간을 기록하여 0.5초마다 ArUco Detection 상태를 전송
            # current_time = time.time()
            # if current_time - last_sent_time >= 0.5:
            #     if ids is not None:
            #         ad_status = 1
            #     else:
            #         ad_status = 0

            #     # AD ArUco Detection 상태가 변했거나 처음인 경우 전송
            #     if detection_status != ad_status or detection_status is None:
            #         ad_status_packet = b'AD' + ad_status.to_bytes(1, byteorder="big")
            #         central_sock.sendall(ad_status_packet + b'\n')
            #         print(f"ArUco Detection 상태 전송 : AD {ad_status}")
            #         detection_status = ad_status

            #     last_sent_time = current_time

            # # 1번, 12번 마커를 인식하면 나무 스캔 시작(1) 또는 종료(0) 메세지 전송
            # if ids is not None:
            #     for marker_id in ids.flatten():
            #         if marker_id == 13 and flag == False:
            #             # 스캔 시작 송신
            #             ts_status_packet = b'TS' + (1).to_bytes(1, byteorder="big")
            #             central_sock.sendall(ts_status_packet + b'\n')
            #             print(f"나무스캔 시작 : TS 1")
            #             flag = True
            #         elif marker_id == 14 and flag == True:
            #             # 스캔 종료 송신
            #             ts_status_packet = b'TS' + (0).to_bytes(1, byteorder="big")
            #             central_sock.sendall(ts_status_packet + b'\n')
            #             print(f"나무스캔 종료 : TS 0")
            #             flag = False

            # ----------------- ArUco Marker Detection & 3D 삼각측량 ------------------- #

            # Annotate frame with marker information
            frame_annotated = rotated_frame_undist.copy()
            if ids is not None and corners is not None:
                aruco.drawDetectedMarkers(frame_annotated, corners, ids)
                rvec, tvec = estimate_camera_pose(corners, ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs)
                if rvec is not None and tvec is not None:
                    # marker_poses = measure_marker_distances(corners, ids, marker_size_cm, camera_matrix, dist_coeffs)
                    marker_poses, average_scale = measure_marker_distances(corners, ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs)
                    if average_scale is not None:
                        tvec *= average_scale
                        for marker_id, data in marker_poses.items():
                            data['tvec'] *= average_scale * 100
                            data['distance'] *= np.linalg.norm(data['tvec'])
                    for marker_id, data in marker_poses.items():
                        distance = data['distance']
                        i = np.where(ids == marker_id)[0][0]
                        center = tuple(np.mean(corners[i][0], axis=0).astype(int))
                        cv.putText(frame_annotated, f"{distance:.2f}cm", (center[0]-30, center[1]-30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,100), 2)
                else:
                    marker_poses = {}
            else:
                marker_poses = {}

            # YOLO 객체 추적 및 중심 좌표 추출
            results = model.track(frame_annotated, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.3, classes=None, verbose=False)
            object_centers = get_object_centers(results)

            # 바운딩 박스와 ID로 객체 텍스트 추가
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    class_name = int(box.cls)

                    # bounding box 그리기
                    cv.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # id 및 class_name 텍스트 추가
                    label = f"ID:{track_id}" if track_id != -1 else "Unknown"
                    # cv.putText(frame_annotated, class_name, (np.mean(x1,x2), np.mean(y1,y2)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv.putText(frame_annotated, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 출력 프레임 표시
            cv.imshow('YOLO Tracking with 3D Triangulation', frame_annotated)

            # Handle Key Presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("프로그램을 종료합니다.")
                break

            elif key == ord('s'):
                if len(marker_poses) == 0:
                    print("ArUco 마커를 감지할 수 없어 캡처할 수 없습니다.")
                    continue

                capture_time = time.strftime("%Y%m%d-%H%M%S")
                # Save the current frame without bounding boxes
                frame_no_bbox = frame_annotated.copy()
                # Remove bounding boxes by redrawing only marker annotations
                # if ids is not None:
                #     aruco.drawDetectedMarkers(frame_no_bbox, corners, ids)
                frame_filename = os.path.join(save_dir, f"capture_{capture_time}.png")
                cv.imwrite(frame_filename, frame_no_bbox)
                print(f"프레임을 저장했습니다: {frame_filename}")

                # Save object centers and camera pose
                saved_frames.append({
                    'time': capture_time,
                    'object_centers': object_centers.copy(),
                    'marker_poses': marker_poses.copy(),
                    'rvec': rvec.copy() if 'rvec' in locals() else None,
                    'tvec': tvec.copy() if 'tvec' in locals() else None,
                    'frame': frame_no_bbox.copy()
                })
                print(f"캡처 {len(saved_frames)}번 완료.")

            elif key == ord('d'):
                if len(saved_frames) < 2:
                    print("두 개 이상의 프레임을 캡처해야 삼각측량을 수행할 수 있습니다.")
                    continue
                else:
                    print("저장된 프레임들을 처리하여 3D 좌표를 계산합니다.")

                    points_3d_marker = triangulate_3d_points(saved_frames, camera_matrix, dist_coeffs)
                    points_3d_marker = calibrate_3d_points(points_3d_marker, marker_positions_cm)

                    # 꽃봉오리(bud) class_num=1, 개화된 꽃(blossom) class_num=2, 인공수분된 꽃 class_num=3에 대해 각각 꽃 개수를 계산합니다.
                    num_bud, unique_bud_points = count_flowers_by_distance(points_3d_marker, distance_threshold=1.0, class_num=1)
                    num_blossom, unique_blossom_points = count_flowers_by_distance(points_3d_marker, distance_threshold=1.0, class_num=2)
                    num_pollination, unique_pollination_points = count_flowers_by_distance(points_3d_marker, distance_threshold=1.0, class_num=3)

                    print(f"꽃봉오리 개수: {num_bud}")
                    print(f"개화된 꽃 개수: {num_blossom}")
                    print(f"인공수분된 꽃 개수: {num_pollination}")

                    
                    print("다음 캡처를 위해 초기화합니다.")
                    saved_frames = []

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("카메라를 해제했습니다.")
        cv.destroyAllWindows()
        print("모든 OpenCV 창을 닫았습니다.")

if __name__ == "__main__":
    main()
