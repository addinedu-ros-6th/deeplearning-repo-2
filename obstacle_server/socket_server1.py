import socket
import struct
import pickle
import cv2
from ultralytics import YOLO

def client(server_ip, port):
    # YOLOv8 모델 불러오기
    model = YOLO('/home/jeback/dev_ws/dl_project/data/obstacle_train_model/best.pt')  # YOLOv8 모델을 로드

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print(f"서버 {server_ip}:{port}에 연결됨")

    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        try:
            # 패킷 헤더(프레임 길이) 수신
            while len(data) < payload_size:
                data += client_socket.recv(4096)
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # 실제 프레임 데이터 수신
            while len(data) < msg_size:
                data += client_socket.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 프레임 디코딩
            frame = pickle.loads(frame_data)

            # YOLOv8 모델로 객체 감지 적용
            results = model(frame)

            # 감지된 결과가 있을 때 처리
            if results and len(results[0].boxes) > 0:            
                for box in results[0].boxes:    
                    # 각 객체에 대해 클래스 ID, 좌표, 신뢰도 점수 추출
                    class_id = int(box.cls)       # 클래스 ID
                    class_name = model.names[class_id]  # 클래스 이름            
                # 감지 결과 데이터를 직렬화(pickle)하여 서버로 전송
                    if class_name == 'elk':
                        message = "OSDS\n".encode()
                        client_socket.sendall(message)  # 감지 결과 전송

            # 수신한 프레임을 화면에 표시 (선택 사항)
            annotated_frame = results[0].plot()
            cv2.imshow("Obstacle_Client", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료 신호 수신")
                break

        except Exception as e:
            print(f"클라이언트 오류 발생: {e}")
            break

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    client("192.168.0.140", 9002)  # 서버 주소와 포트