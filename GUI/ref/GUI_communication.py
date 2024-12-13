import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5 import uic
import PyQt5
from PyQt5.QtCore import Qt, QThread, pyqtSignal
# resources_rc 파일은 icon, image, qss 파일을 로드하기 위한 파일
import resources_rc
import socket  # 통신 소켓 모듈 추가

# 아래 if문 2개는 고해상도 모니터에서 GUI 화면이 작게 보이는 문제 해결하기 위한 코드
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class MessageReceiver(QThread):
    # 메시지를 받으면 시그널로 보내기
    received = pyqtSignal(bytes)

    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.running = True

    def run(self):
        try:
            while self.running:
                header = b''
                while len(header) < 2:
                    header += self.sock.recv(2 - len(header))

                if header == b'RC':
                    data = self.sock.recv(2)
                    print(f"receive : {data}")
                    datan= int.from_bytes(data[:1], byteorder="big")
                    print(f"datan : {datan}")
                    # if datan == 0:
                    #     self.pushButton_5.setEnabled(True)
        except Exception as e:
            print(f"메시지 수신 오류: {e}")

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()

        # .ui 파일을 로드
        uic.loadUi(r"C:\Users\dldms\Documents\applecare\deeplearning-repo-2-dev\GUI\new_GUI.ui", self)
        
        # "192.168.1.11" 포트 1234로 메시지 전송
        self.server_ip = "192.168.0.134"
        self.server_port = 1234
        
        # TCP 소켓 생성
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.server_port))  # 서버에 연결

        # Btn_Schedule_Setting 클릭 시 페이지 전환 이벤트 연결
        self.Btn_Main_Dashboard.clicked.connect(self.show_main_dashboard)
        self.Btn_Pollination_Monitor.clicked.connect(self.show_pollination_monitor)
        self.Btn_Statistics.clicked.connect(self.show_statistics)
        self.Btn_Schedule_Setting.clicked.connect(self.show_schedule_setting)

        # pushButton_5 클릭 시 메세지 전송 연결
        self.pushButton_5.clicked.connect(self.send_message)

        # MessageReceiver 스레드 생성 및 시작
        self.receiver_thread = MessageReceiver(self.sock)
        self.receiver_thread.received.connect(self.process_received_message)
        # self.receiver_thread.received.connect(self.handle_received_message)
        self.receiver_thread.start()
        # self.receive_message()

    def show_main_dashboard(self):
        print("Btn_Main_Dashboard clicked")
        self.stackedWidget.setCurrentWidget(self.Main_Dashboard_2)

    def show_pollination_monitor(self):
        print("Btn_Pollination_Monitor clicked")
        self.stackedWidget.setCurrentWidget(self.Pollination_Monitor_2)

    def show_statistics(self):
        print("Btn_Statistics clicked")
        self.stackedWidget.setCurrentWidget(self.Statistics_2)
    
    def show_schedule_setting(self):
        print("Btn_Schedule_Setting clicked")
        self.stackedWidget.setCurrentWidget(self.Schedule_Setting_2)

    def process_received_message(self, data):
        """수신된 데이터를 처리하는 함수"""
        print(f"수신한 데이터: {data}")
        # 여기서 데이터를 이용한 GUI 업데이트 또는 추가 처리를 구현합니다.
        # 예시: 수신한 데이터가 0이면 pushButton_5를 활성화
        if int.from_bytes(data[:1], byteorder="big") == 0:
            self.pushButton_5.setEnabled(True)

    def send_message(self):
        # # "192.168.1.11" 포트 1234로 메시지 전송
        # self.server_ip = "192.168.0.11"
        # self.server_port = 1234
        command = 1
        start = b"RC"
        send_data = start + command.to_bytes(1, byteorder="big") + b'\n'

        try:
            # TCP 소켓 생성
            # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # sock.connect((self.server_ip, self.server_port))  # 서버에 연결

            # 메시지 전송
            # sock.sendall(command.encode('utf-8'))
            self.sock.sendall(send_data)
            print(f"메시지 '{send_data}'가 {self.server_ip}:{self.server_port}로 전송되었습니다.")

            self.pushButton_5.setEnabled(False)
            
            # 연결 종료
            # self.sock.close()
        except Exception as e:
            print(f"메시지 전송 오류: {e}")

    # def receive_message(self):
    #     try:
    #         while True:
    #             header = b''
    #             while len(header) < 2:
    #                 header += self.sock.recv(2 - len(header))

    #                 if header == b'RC':
    #                     data = self.sock.recv(2)
    #                     print(f"receive : {data}")
                        
    #                     if int.to_bytes(data[:1], byteorder="big") == 0:
    #                         self.pushButton_5.setEnabled(True)

    #     except Exception as e:
    #         print(f"메시지 수신 오류: {e}")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
