import cv2 as cv
from cv2 import aruco
import numpy as np
import os
import time
from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' for non-GUI rendering if 'TkAgg' isn't available

# ----------------------------- Helper Functions ----------------------------- #

def draw_rotated_text(img, text, position, angle, font=cv.FONT_HERSHEY_PLAIN, 
                     scale=1.3, color=(0, 0, 255), thickness=2):
    """
    Draws text rotated by a specified angle on an image with boundary checks.
    """
    # Get the text size
    text_size, baseline = cv.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size

    # Create a transparent image for the text
    padding = 10
    text_img = np.zeros((text_height + baseline + padding, text_width + padding, 4), dtype=np.uint8)
    
    # Put the text onto the transparent image
    cv.putText(
        text_img, 
        text, 
        (padding // 2, text_height + baseline), 
        font, 
        scale, 
        color + (255,),  # Add alpha channel
        thickness, 
        cv.LINE_AA
    )
    
    # Rotate the text image
    M = cv.getRotationMatrix2D((text_img.shape[1]//2, text_img.shape[0]//2), angle, 1)
    rotated_text = cv.warpAffine(
        text_img, 
        M, 
        (text_img.shape[1], text_img.shape[0]), 
        flags=cv.INTER_LINEAR, 
        borderMode=cv.BORDER_CONSTANT, 
        borderValue=(0,0,0,0)
    )
    
    # Get the region of interest on the original image
    x, y = position
    h, w, _ = rotated_text.shape
    img_h, img_w, _ = img.shape

    # Adjust x and y to ensure they are within the image boundaries
    if x < 0:
        rotated_text = rotated_text[:, -x:]
        w += x  # x is negative
        x = 0
    if y < 0:
        rotated_text = rotated_text[-y:, :]
        h += y  # y is negative
        y = 0

    # Adjust width and height if the text exceeds image boundaries
    if x + w > img_w:
        w = img_w - x
        rotated_text = rotated_text[:, :w]
    if y + h > img_h:
        h = img_h - y
        rotated_text = rotated_text[:h, :]

    # After adjustments, ensure h and w are positive
    if h <= 0 or w <= 0:
        print(f"Skipped drawing text '{text}' due to boundary issues.")
        return  # Cannot draw text outside the image

    # Extract the ROI from the original image
    roi = img[y:y+h, x:x+w]

    # Create a mask from the alpha channel of the rotated text
    mask = rotated_text[:, :, 3] / 255.0
    inv_mask = 1.0 - mask

    # Blend the rotated text with the ROI
    for c in range(0, 3):
        roi[:, :, c] = (mask * rotated_text[:, :, c] + inv_mask * roi[:, :, c])

    # Replace the ROI on the original image with the blended result
    img[y:y+h, x:x+w] = roi

# ----------------------------- Helper Functions ----------------------------- #

# def detect_aruco_markers(image, dictionary, parameters):
    """
    Detects Aruco markers in the image and returns their IDs and corner points.
    """
    # corners, ids, rejected = aruco.detectMarkers(image, dictionary, parameters=parameters)
    # if ids is not None:
    #     # Convert each marker's corners to 2D coordinates
    #     all_corners = []
    #     all_ids = []
    #     for marker_corners, marker_id in zip(corners, ids.flatten()):
    #         # marker_corners is (1, 4, 2) -> reshape to (4, 2)
    #         reshaped_corners = marker_corners.reshape(-1, 2).astype(np.float32)
    #         if reshaped_corners.shape != (4, 2):
    #             print(f"Invalid marker corners shape: {reshaped_corners.shape}")
    #             continue
    #         all_corners.append(reshaped_corners)  # This causes the shape to be (4, 2)
    #         all_ids.append(marker_id)
    #     print(f"Detected {len(all_ids)} ArUco marker(s).")
    #     for idx, marker_id in enumerate(all_ids):
    #         print(f"Marker ID: {marker_id}, Corners: {all_corners[idx]}")
    #     return np.array(all_ids), all_corners  # corners are now list of (4,2) arrays
    # else:
    #     print("No ArUco markers detected.")
    # return None, None

def draw_markers_with_shifted_text(frame, marker_corners, marker_IDs, shift_x, shift_y):
    try:
        # 마커 그리기 (ID 텍스트는 생략)
        # cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs, borderColor=(0, 255, 0))

        # 각 마커의 ID를 수동으로 표시
        if marker_IDs is not None:
            for i, marker_corner in enumerate(marker_corners):
                # 마커의 좌측 상단 좌표 가져오기
                top_left_corner = marker_corner[0][0]  # marker_corner는 (4, 2) shape의 배열임
                marker_id = marker_IDs[i][0]  # 각 마커의 ID
                
                # 텍스트 위치 계산 (좌측 상단에서 shift_x, shift_y만큼 이동)
                text_position = (int(top_left_corner[0] + shift_x), int(top_left_corner[1] + shift_y))
                
                # ID 텍스트 그리기
                cv.putText(frame, f"ID: {marker_id}", text_position, cv.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1 , cv.LINE_AA)
    except cv.error as e:
        print(f"OpenCV error during drawDetectedMarkers: {e}")
    
    return frame

def detect_aruco_markers(image, dictionary, parameters):
    """
    Detects Aruco markers in the image and returns their IDs and corner points.
    """
    corners, ids, rejected = aruco.detectMarkers(image, dictionary, parameters=parameters)
    if ids is not None:
        # print(f"Detected {len(ids)} ArUco marker(s).")
        for idx, marker_id in enumerate(ids.flatten()):
            pass
            # print(f"Marker ID: {marker_id}, Corners: {corners[idx]}")
        return ids, corners  # Return corners as is without reshaping
    # else:
    #     print("No ArUco markers detected.")
    return None, None

def annotate_markers(frame, marker_IDs, marker_corners, marker_size, camera_matrix, dist_coeffs, draw_text_func, text_angle=0):
    """
    Annotates detected ArUco markers on the frame by drawing their boundaries and estimating camera pose.
    Returns the annotated frame and estimated rotation and translation vectors.
    """
    if marker_corners is None or marker_IDs is None:
        # print("No markers to annotate.")
        return frame, None, None  # No markers to annotate

    # print(f"Annotating {len(marker_IDs)} marker(s).")
    # print(f"Type of marker_corners: {type(marker_corners)}")
    # print(f"Shapes of marker_corners elements: {[corner.shape for corner in marker_corners]}")
    # print(f"Dtypes of marker_corners elements: {[corner.dtype for corner in marker_corners]}")  # Added line

    # Define 3D points of the marker corners in the marker's coordinate system
    # Assuming markers are placed on the XY plane with Z=0
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    # Lists to store object points and image points for solvePnP
    obj_points_all = []
    img_points_all = []

    for i in range(len(marker_IDs)):
        corners = marker_corners[i]
        reshaped_corners = corners.reshape(-1, 2)
        obj_points_all.append(obj_points)
        img_points_all.append(reshaped_corners)

    obj_points_all = np.vstack(obj_points_all)
    img_points_all = np.vstack(img_points_all)

    # print(f"Object Points Shape: {obj_points_all.shape}")
    # print(f"Image Points Shape: {img_points_all.shape}")

    # Ensure the points are of type float32 or float64
    if obj_points_all.dtype != np.float32 and obj_points_all.dtype != np.float64:
        obj_points_all = obj_points_all.astype(np.float32)
    if img_points_all.dtype != np.float32 and img_points_all.dtype != np.float64:
        img_points_all = img_points_all.astype(np.float32)

    # Estimate pose using solvePnP
    success, rvec, tvec = cv.solvePnP(obj_points_all, img_points_all, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    if not success:
        print("Pose estimation failed.")
        return frame, None, None

    # Correct function call to draw detected markers
    # print("Drawing detected markers.")
    try:
        draw_markers_with_shifted_text(frame, marker_corners, marker_IDs, shift_x=10, shift_y=-10)
        # cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs)
    except cv.error as e:
        print(f"OpenCV error during drawDetectedMarkers: {e}")
        return frame, rvec, tvec

    for i in range(len(marker_IDs)):
        # Draw the pose axes
        # cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, int(marker_size * 100))  # Adjust axis length as needed

        # Optionally, add text with marker ID
        center = tuple(np.mean(marker_corners[i][0], axis=0).astype(int))
        text = f"ID:{marker_IDs[i]}"
        # draw_rotated_text(frame, text, (center[0], center[1]), angle=text_angle)

    return frame, rvec, tvec

def get_object_centers(results):
    """
    Extracts the center coordinates of detected objects from YOLO results.
    Returns a dictionary with object IDs as keys and center coordinates as values.
    """
    centers = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # 객체 ID
            track_id = int(box.id[0]) if box.id is not None else -1
            
            if track_id not in centers:
                centers[track_id] = []
            centers[track_id].append((track_id, (center_x, center_y)))
    return centers

def estimate_camera_pose(marker_corners, marker_ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs):
    """
    검출된 ArUco 마커와 그들의 알려진 위치를 사용하여 카메라 자세를 추정합니다.
    """
    object_points = []
    image_points = []
    half_size = marker_size_cm / 2.0
    for i, marker_id in enumerate(marker_ids.flatten()):
        if marker_id in marker_positions_cm:
            # 마커의 중심 위치 가져오기
            center_pos = marker_positions_cm[marker_id]
            # 마커의 코너에 대한 객체 포인트 정의
            obj_pts = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
            # 마커의 위치로 코너 이동
            obj_pts += center_pos.reshape(1, 3)
            object_points.append(obj_pts)
            # 이미지 포인트 가져오기 (마커의 코너)
            img_pts = marker_corners[i][0].reshape(4, 2)
            image_points.append(img_pts)
        else:
            print(f"Marker ID {marker_id} not in known marker positions.")
    
    if len(object_points) == 0:
        print("No known markers detected for pose estimation.")
        return None, None
    
    object_points = np.concatenate(object_points, axis=0).astype(np.float32)
    image_points = np.concatenate(image_points, axis=0).astype(np.float32)
    
    # 단위 일관성을 위해 object_points를 미터 단위로 변환 (cm -> m)
    object_points_m = object_points
    
    success, rvec, tvec = cv.solvePnP(object_points_m, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    
    if not success:
        print("Camera pose estimation failed.")
        return None, None
    
    return rvec, tvec

def measure_marker_distances(marker_corners, marker_ids, marker_size, camera_matrix, dist_coeffs):
    """
    Measures the distance to each detected ArUco marker.
    Returns a dictionary mapping marker IDs to their distances and poses.
    """
    distances = {}
    for i in range(len(marker_ids)):
        corners = marker_corners[i]
        marker_id = marker_ids[i][0]  # marker_ids is [[id1], [id2], ...]
        # Define object points for the marker in its own coordinate system
        obj_points = np.array([
            [-marker_size/2, marker_size/2, 0],
            [marker_size/2, marker_size/2, 0],
            [marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
        # Image points
        img_points = corners.reshape(-1, 2)
        # Estimate pose
        success, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if success:
            # Compute distance as norm of tvec
            distance = np.linalg.norm(tvec)
            distances[marker_id] = {'distance': distance, 'rvec': rvec, 'tvec': tvec}
        else:
            print(f"Pose estimation failed for marker ID {marker_id}")
    return distances

def calibrate_3d_points(points_3d_marker, marker_positions_cm):
    """
    삼각측량된 3D 포인트를 ArUco 마커의 실제 위치를 기반으로 보정합니다.
    간단한 스케일 보정 예시.
    """
    # ArUco 마커의 ID와 실제 위치가 알려진 경우
    # 여기서는 예시로 하나의 마커를 사용하여 스케일을 보정합니다.
    # 실제로는 여러 마커를 사용하여 더 정밀한 보정을 할 수 있습니다.
    
    if 14 in points_3d_marker and 14 in marker_positions_cm:
        # 마커 ID 14의 삼각측량된 포인트와 실제 위치
        triangulated_point = points_3d_marker[14]
        actual_position = marker_positions_cm[14]
        
        # 스케일 팩터 계산 (실제 거리 / 삼각측량된 거리)
        triangulated_distance = np.linalg.norm(triangulated_point)
        actual_distance = np.linalg.norm(actual_position)
        scale_factor = actual_distance / triangulated_distance
        
        print(f"스케일 팩터: {scale_factor}")
        
        # 모든 3D 포인트에 스케일 팩터 적용
        for obj_id in points_3d_marker:
            points_3d_marker[obj_id] *= scale_factor
    
    return points_3d_marker

def remove_duplicates(points_3d_list, distance_threshold=5.0):
    """
    리스트에서 중복되는 3D 포인트를 제거합니다.
    포인트 간의 거리가 distance_threshold(cm) 미만이면 중복으로 간주합니다.
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
    try:
        # 1. Parse Command-Line Arguments
        parser = argparse.ArgumentParser(description='YOLO Object Tracking with 3D Triangulation using Epipolar Geometry')
        parser.add_argument('--model', type=str, default='/home/ask/OneDrive/Document/dev_ws_DL/count_flower/best_90.pt',
                            help='Path to the YOLOv8 model file')
        parser.add_argument('--calib_data', type=str, default='/home/ask/OneDrive/Document/dev_ws_DL/count_flower/logi_calibration.npz',
                            help='Path to the camera calibration data (.npz file)')
        parser.add_argument('--save_dir', type=str, default='captured_frames/home/ask/OneDrive/Document/dev_ws_DL/count_flower/5.YOLO_coordinate/save_dir',
                            help='Directory to save captured frames and data')
        parser.add_argument('--marker_size_cm', type=float, default=3.5,  # in centimeters
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
        video_devices = []
        for device in os.listdir('/dev/'):
            if device.startswith('video'):
                video_devices.append(device)
        # print("Available video devices:", video_devices)
        for device in video_devices:
            device_id = int(device.replace('video', ''))
            cap = cv.VideoCapture(device_id)
            if cap.isOpened() and device_id != 0:
                print(f"Device /dev/{device} (ID: {device_id}) is available.")
                cap.release()
                external_cam = device_id
        
        cap = cv.VideoCapture(external_cam)  # 기본 웹캠 사용
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        print("웹캠이 성공적으로 열렸습니다.")

        # 5. Prepare Save Directory
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"저장 디렉토리를 생성했습니다: {save_dir}")
        else:
            print(f"저장 디렉토리가 이미 존재합니다: {save_dir}")

        # 6. Set Up ArUco Marker Detector
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        parameters = aruco.DetectorParameters()
        # print("ArUco 마커 검출기가 설정되었습니다.")

        # 7. Known Marker Positions (in cm)
        marker_positions_cm = {
            13: np.array([100.0, 0.0, 100.0], dtype=np.float32),
            14: np.array([0.0, 50.0, 0.0], dtype=np.float32),
            # 필요한 만큼 마커 추가
        }
        marker_size_cm = args.marker_size_cm

        # 8. Initialize Variables for Captures
        captures = []  # List to store capture data
        capture_count = 0
        point_3d_list = []
        
        # 9. Main Processing Loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # Undistort Frame
            frame_undist = cv.undistort(frame, camera_matrix, dist_coeffs)
            rotated_frame_undist = cv.rotate(frame_undist, cv.ROTATE_90_COUNTERCLOCKWISE)

            # Detect ArUco Markers and Estimate Pose
            gray = cv.cvtColor(rotated_frame_undist, cv.COLOR_BGR2GRAY)
            ids, corners = detect_aruco_markers(gray, aruco_dict, parameters)
            frame_annotated = rotated_frame_undist.copy()
            if ids is not None and corners is not None:
                rvec, tvec = estimate_camera_pose(corners, ids, marker_positions_cm, marker_size_cm, camera_matrix, dist_coeffs)
                if rvec is None or tvec is None:
                    print("Camera pose estimation failed.")
                    continue
                marker_poses = measure_marker_distances(corners, ids, args.marker_size_cm, camera_matrix, dist_coeffs)
                for marker_id, data in marker_poses.items():
                    distance = data['distance']
                    # print(f"Distance to marker ID {marker_id}: {distance:.2f} meters")
                    # Annotate distance on the frame
                    i = np.where(ids == marker_id)[0][0]
                    center = tuple(np.mean(corners[i][0], axis=0).astype(int))
                    cv.putText(frame_annotated, f"{distance:.2f}cm", center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            else:
                marker_poses = {}
                # print("No markers detected.")
                rvec, tvec = None, None

            # Annotate Markers
            if ids is not None and corners is not None:
                draw_markers_with_shifted_text(frame_annotated, corners, ids, shift_x=10, shift_y=-10)

            # Perform YOLO Object Tracking
            results = model.track(frame_annotated, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.3, classes=None, verbose=False)

            # Extract Object Centers
            object_centers = get_object_centers(results)

            # Annotate Objects with Bounding Boxes and IDs
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Object ID
                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Draw bounding box
                    cv.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add text (ID)
                    label = f"ID:{track_id}" if track_id != -1 else "Unknown"
                    cv.putText(frame_annotated, label, (x1, y1 - 10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the Annotated Frame
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
                if ids is not None:
                    aruco.drawDetectedMarkers(frame_no_bbox, corners, ids)
                frame_filename = os.path.join(save_dir, f"capture_{capture_time}.png")
                cv.imwrite(frame_filename, frame_no_bbox)
                print(f"프레임을 저장했습니다: {frame_filename}")

                # Save object centers and camera pose
                captures.append({
                    'time': capture_time,
                    'object_centers': object_centers.copy(),
                    'rvec': rvec.copy(),
                    'tvec': tvec.copy(),
                    'frame': frame_no_bbox.copy()
                })
                capture_count += 1
                print(f"캡처 {capture_count}번 완료.")

                # If two captures are done, perform triangulation
                if capture_count == 2:
                    print("두 번의 캡처가 완료되었습니다. 3D 좌표를 계산합니다.")
                    capture1, capture2 = captures

                    # Projection matrices
                    R1, _ = cv.Rodrigues(capture1['rvec'])
                    t1 = capture1['tvec']
                    R2, _ = cv.Rodrigues(capture2['rvec'])
                    t2 = capture2['tvec']

                    # Ensure t1 and t2 are (3, 1)
                    t1 = t1.reshape(3, 1)
                    t2 = t2.reshape(3, 1)

                    P1 = camera_matrix @ np.hstack((R1, t1))
                    P2 = camera_matrix @ np.hstack((R2, t2))
                    
                    # Convert object_centers to dictionaries for easy lookup
                    centers1 = {obj_id: pos for obj_id, pos in capture1['object_centers'].items()}
                    centers2 = {obj_id: pos for obj_id, pos in capture2['object_centers'].items()}

                    # Find common object IDs
                    common_ids = set(centers1.keys()).intersection(centers2.keys())

                    if not common_ids:
                        print("두 캡처에서 공통된 ArUco 마커가 없어 삼각측량을 수행할 수 없습니다.")
                        captures = []
                        capture_count = 0
                        continue
                    else:
                        print(f"공통된 객체 ID: {common_ids}")

                        # Triangulate 3D points for each common ID
                        points_3d_marker = {}
                        for obj_id in common_ids:
                            pt1 = centers1[obj_id][0][1]
                            pt2 = centers2[obj_id][0][1]
                            print(f"Triangulating for ID {obj_id}: Capture1 (x={pt1[0]}, y={pt1[1]}), Capture2 (x={pt2[0]}, y={pt2[1]})")
                            point_4d_hom = cv.triangulatePoints(P1, P2, np.array(pt1).reshape(2,1), np.array(pt2).reshape(2,1))
                            point_3d = cv.convertPointsFromHomogeneous(point_4d_hom.T)[0][0]  # (x, y, z)
                            # 좌표를 cm 단위로 변환 (이미 cm 단위이므로 그대로 사용)
                            points_3d_marker[obj_id] = point_3d

                        # 보정 과정 추가
                        points_3d_marker = calibrate_3d_points(points_3d_marker, marker_positions_cm)

                        # Remove duplicates
                        # points_3d_list = list(points_3d_marker.values())
                        # unique_points = remove_duplicates(points_3d_list, distance_threshold=2.0)  # 2cm 이하인 경우 중복으로 간주
                        unique_points = list(points_3d_marker.values())
                        
                        # Print 3D Points
                        print("\nTriangulated 3D Points in Marker Coordinate System:")
                        for obj_id, point in points_3d_marker.items():
                            print(f"ID: {obj_id}, 3D Point: {point}")
                            point_3d_list.append(point)

                        # Count the number of unique flowers
                        num_flowers = len(unique_points)
                        print(f"\n꽃의 개수는 {num_flowers}개 입니다.")

                        # Visualize 3D Points by saving to a file
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        for obj_id, point in zip(points_3d_marker.keys(), unique_points):
                            ax.scatter(point[0], point[1], point[2], label=f'ID:{obj_id}')
                            ax.text(point[0], point[1], point[2], f'ID:{obj_id}')
                        ax.set_xlabel('X (cm)')
                        ax.set_ylabel('Y (cm)')
                        ax.set_zlabel('Z (cm)')
                        ax.legend()
                        plot_filename = os.path.join(save_dir, f"triangulated_3d_points_{capture_time}.png")
                        plt.savefig(plot_filename)
                        plt.close(fig)
                        print(f"3D plot saved as {plot_filename}")

                    # Reset captures for next triangulation
                    captures = []
                    capture_count = 0
                    print("삼각측량을 완료했습니다. 다음 캡처를 위해 초기화합니다.")              

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            print("카메라를 해제했습니다.")
            print("avg, std of 3D points: ")
            values_1, values_2, values_3 = [], [], []
            averages, std_devs = [], []
            for i in range(3):
                point_3d_list = np.array(point_3d_list)
                # 3D 좌표 리스트에서 각 축별로 값을 분리하여 저장
            for point in point_3d_list:
                values_1.append(point[0])  # x 축 값
                values_2.append(point[1])  # y 축 값
                values_3.append(point[2])  # z 축 값
            print("values_1 : ", values_1)
            print("values_2 : ", values_2)
            print("values_3 : ", values_3)
            # 각 축에 대해 평균과 표준편차 계산
            for values in [values_1, values_2, values_3]:
                avg_value = np.mean(values)
                std_value = np.std(values)
                averages.append(avg_value)
                std_devs.append(std_value)
            print("averages : ", averages)
            print("std_devs : ", std_devs)

            # 출력
            for i, (avg, std) in enumerate(zip(averages, std_devs)):
                print(f"avg_{i}: {avg}, std_{i}: {std}")
            
        cv.destroyAllWindows()
        print("모든 OpenCV 창을 닫았습니다.")

if __name__ == "__main__":
    main()
