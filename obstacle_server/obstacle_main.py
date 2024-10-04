import cv2, socket, struct
import numpy as np
from ultralytics import YOLO

# central server ip, port
central_server_ip = "192.168.0.23"
central_server_port = 4040

central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    central_server_socket.connect((central_server_ip, central_server_port))
    print(f"{central_server_ip}:{central_server_port} 서버에 연결되었습니다.")
except Exception as e:
    print(f"{central_server_ip}:{central_server_port} 서버 연결 오류")
    central_server_socket = None

model = YOLO("/home/jh/project/deeplearning_project/deeplearning-repo-2/obstacle_server/best.pt")

try:
    while True:
        head = b''

        while len(head) < 6:
            head += central_server_socket.recv(6 - len(head))
        print(f"head: {head}")
        if head[:2] == b'SF':
            frame_size = struct.unpack(">L", head[2:6])[0]
            frame_data = b''

            while len(frame_data) < frame_size:
                packet = central_server_socket.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet
            
            # end data b'\n' 수신
            central_server_socket.recv(1)
            if len(frame_data) > 0:
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                results = model(frame)
                annotated_frame = results[0].plot()

                detections = results[0].boxes
                detection_data = []
                for box in detections:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detection_data.append((class_id, x1, y1, x2, y2))
                print(f"results: {detection_data}")

                if frame is not None:
                    cv2.imshow("detect frame", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                print("Error: Unable to decode frame")
except Exception as e:
    print(f"Error receiving or displaying frame: {e}")

finally:
    central_server_socket.close()
    cv2.destroyAllWindows()