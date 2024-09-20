import cv2, socket, struct
import numpy as np

# pollination server ip, port
server_ip = "192.168.0.50"
server_port = 3141

# 소켓 생성 및 바인딩
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)
print(f"서버가 {server_ip} : {server_port}에서 대기 중입니다...")

# 클라이언트 연결 수립
conn, addr = server_socket.accept()
print(f"클라이언트 {addr}와 연결되었습니다.")

try:
    while True:
        header = b''

        while len(header) < 2:
            header += conn.recv(2 - len(header))
        
        # frame data
        if header == b'SF':
            size_data = b''
            while len(size_data) < 4:
                packet = conn.recv(4 - len(size_data))
                if not packet:
                    break
                size_data += packet
            frame_size = struct.unpack(">L", size_data)[0]
            
            frame_data = b''
            while len(frame_data) < frame_size:
                packet = conn.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet

            conn.recv(1)
            
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow("received frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Unable to decode frame")
        
        # status data
        elif header == b'CS':
            status_data = b''
            while len(status_data) < 1:
                packet = conn.recv(1 - len(status_data))
                if not packet:
                    break
                status_data += packet
            
            conn.recv(1)

            status = int.from_bytes(status_data, byteorder="big")
            if status == 1:
                print("status:", status)
            else:
                print("status", status)

except Exception as e:
    print(f"Error receiving or displaying frame: {e}")

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
