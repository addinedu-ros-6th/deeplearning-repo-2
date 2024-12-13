import cv2
import socket
import struct
import os
import threading
import time

# status
# before_status = 0

# Central server details
CENTRAL_SERVER_IP = "192.168.0.13"
CENTRAL_SERVER_PORT = 3141

# Pollination server details (if needed)
POLLINATION_SERVER_IP = "192.168.0.42"
POLLINATION_SERVER_PORT = 9003

def connect_to_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((ip, port))
        print(f"Connected to server at {ip}:{port}")
    except Exception as e:
        print(f"Error connecting to server {ip}:{port} - {e}")
        sock = None
    return sock

def send_frame(sock, cam):
    while True:
        try:
            # Capture frame from /dev/video0 or /dev/video1
            ret, frame = cam.read()
            if ret and sock:
                # Encode frame from /dev/video0 or /dev/video1
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                frame_size = len(frame_data)
                header = b'SF'
                sock.sendall(header + struct.pack(">L", frame_size) + frame_data + b'\n')
                print(f"Sent frame of size {frame_size} to {sock.getpeername()}")
        except Exception as e:
            print(f"Error capturing or sending {cam} frame: {e}")
        # Limit to 30 frames per second
        # time.sleep(1/30)

def send_status(sock_central):
    while True:
        try:
            # Check for device availability
            arduino_avail = os.path.exists('/dev/ttyArduino')
            vid0_avail = os.path.exists('/dev/video0')
            vid1_avail = os.path.exists('/dev/video1')
            status = 1 if arduino_avail and vid0_avail and vid1_avail else 0
        
            status_packet = b'CS' + status.to_bytes(1, byteorder="big")
            # status_packet = struct.pack('2sh', b'CS', status)
            sock_central.sendall(status_packet + b'\n')
            print(f"Sent status: CS {status}")
        except Exception as e:
            print(f"Error sending status: {e}")
        time.sleep(5)  # Send status every 10 seconds

def main():
    # Initialize cameras
    cam0 = cv2.VideoCapture(1)
    cam1 = cv2.VideoCapture(0)

    # Connect to central server
    central_sock = connect_to_server(CENTRAL_SERVER_IP, CENTRAL_SERVER_PORT)
    if not central_sock:
        return # Exit if connection fails

    # Connect to pollination server
    pollination_sock = connect_to_server(POLLINATION_SERVER_IP, POLLINATION_SERVER_PORT)
    if not pollination_sock:
        return

    # Create threads
    frame0_thread = threading.Thread(target = send_frame, args = (central_sock, cam0))
    frame1_thread = threading.Thread(target = send_frame, args = (pollination_sock, cam1))
    status_thread = threading.Thread(target = send_status, args = (central_sock,))

    # Start threads
    frame0_thread.start()
    frame1_thread.start()
    status_thread.start()

    try:
        frame0_thread.join()
        frame1_thread.join()
        status_thread.join()
    
    except KeyboardInterrupt:
        print("Interrupted by user")

if __name__ == "__main__":
    main()