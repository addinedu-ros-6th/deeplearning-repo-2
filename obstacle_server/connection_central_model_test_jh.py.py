import socket
import struct
import cv2
import numpy as np
import os
from ultralytics import YOLO

central_server_ip = "192.168.0.13"
central_server_port = 4040

central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
central_server_socket.connect((central_server_ip, central_server_port))
print(f"Connected to server at {central_server_ip}:{central_server_port}")

try:
    while True:
        header = b''
        
        while len(header) < 2:
            packet = central_server_socket.recv(2 - len(header))
        print(f"Received header: {header}")

        
        # frame data
        if header == b'SF':
            size_data = b''
            print("Receiving frame size...")
            while len(size_data) < 4:
                packet = central_server_socket.recv(4 - len(size_data))
                if not packet:
                    break
                size_data += packet
            frame_size = struct.unpack(">L", size_data)[0]
            print(f"frame size: {frame_size}")

            frame_data = b''
            while len(frame_data) < frame_size:
                packet = central_server_socket.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet
            print(f"frmae_data : {frame_data}")
            
            central_server_socket.recv(1)

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                print("received")
                cv2.imshow("received frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Unable to decode frame")

except Exception as e:
    print(f"Error receiving or displaying frame: {e}")

finally:
    cv2.destroyAllWindows()