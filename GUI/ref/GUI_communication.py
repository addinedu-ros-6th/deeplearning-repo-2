import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5 import uic
import PyQt5
from PyQt5.QtCore import Qt
# resources_rc 파일은 icon, image, qss 파일을 로드하기 위한 파일
import resources_rc
import socket  # 통신 소켓 모듈 추가

# 아래 if문 2개는 고해상도 모니터에서 GUI 화면이 작게 보이는 문제 해결하기 위한 코드
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()

        # .ui 파일을 로드
        uic.loadUi("/home/ask/OneDrive/Document/dev_ws_DL/GUI/new_GUI.ui", self)

        # Btn_Schedule_Setting 클릭 시 페이지 전환 이벤트 연결
        self.Btn_Main_Dashboard.clicked.connect(self.show_main_dashboard)
        self.Btn_Pollination_Monitor.clicked.connect(self.show_pollination_monitor)
        self.Btn_Statistics.clicked.connect(self.show_statistics)
        self.Btn_Schedule_Setting.clicked.connect(self.show_schedule_setting)

        # pushButton_5 클릭 시 메세지 전송 연결
        self.pushButton_5.clicked.connect(self.send_message)

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

    def send_message(self):
        # "192.168.1.11" 포트 1234로 메시지 전송
        server_ip = "192.168.1.11"
        server_port = 1234
        message = "1"

        try:
            # TCP 소켓 생성
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((server_ip, server_port))  # 서버에 연결

            # 메시지 전송
            sock.sendall(message.encode('utf-8'))
            print(f"메시지 '{message}'가 {server_ip}:{server_port}로 전송되었습니다.")
            
            # 연결 종료
            sock.close()
        except Exception as e:
            print(f"메시지 전송 오류: {e}")

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
