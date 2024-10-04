import cv2, socket, struct, pandas, json
import numpy as np

# pollination server ip, port
pollination_server_ip = "192.168.45.244"
pollination_server_port = 9003

# 소켓 생성 및 바인딩
pollination_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
pollination_server_socket.bind((pollination_server_ip, pollination_server_port))
pollination_server_socket.listen(1)
print(f"서버가 {pollination_server_ip} : {pollination_server_port}에서 대기 중입니다...")

# 클라이언트 연결 수립
pollination_conn, pollination_addr = pollination_server_socket.accept()
print(f"클라이언트 {pollination_addr}와 연결되었습니다.")

try:
    while True:
        head = b''

        while len(head) < 6:
            head += pollination_conn.recv(6 - len(head))
        
        if head[:2] == b'SF':
            frame_size = struct.unpack(">L", head[2:6])[0]
            frame_data = b''

            while len(frame_data) < frame_size:
                packet = pollination_conn.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet

            pollination_conn.recv(1)
            
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            if frame is not None:
                cv2.imshow("received frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Unable to decode frame")

except Exception as e:
    print(f"Error receiving or displaying frame: {e}")

finally:
    pollination_conn.close()
    pollination_server_socket.close()
    cv2.destroyAllWindows()
