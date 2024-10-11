# 딥러닝 프로젝트 2조. 사과 과수원 모니터링 서비스
> ***DeepLearning Project Team 2 Repository: Orchard Monitoring Service for Pollination and Fruit Set Yield of Apple Flowers***

## 🏁 개요

### 🗓️ 프로젝트 기간
- 9월 9일 - 9월 25일

### 💡 선정 배경 및 프로젝트 개요 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/user-attachments/assets/47b99d2d-8f81-41ab-99a5-99b11ad04f2b"  width="600" height="300"/>

- 노동집약적 산업인 농업 분야에서 고령화가 진행되고 있고 이로인해서 로봇을 이용한 자동화 시스템의 필요성이 커지고 있습니다.
- 그래서 과수원에서 기르는 작물들의 관리를 효율적이고 체계적으로 하기위해 딥러닝을 이용하여 모니터링 할 수 있는 시스템을 구축하고자 하였습니다.
- 그 중 사과나무의 인공수분, 착과량 모니터링 서비스에 집중하였고 주요 핵심기능으로는 과수원 주행 모니터링, 사과 꽃 상태인지, 모니터링 통계기능 등이 있습니다.

### 🛠️ 기술 스택 
|분류|기술|
|-----|-----|
|개발 환경|![js](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=Ubuntu&logoColor=white) |
|개발 언어|![js](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![js](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)|
|사용 기술| <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV"> <img src="https://img.shields.io/badge/PyQt5-21C25E?style=for-the-badge&logo=quicktype"> |
|DBMS|![js](https://img.shields.io/badge/MySQL-00000F?style=for-the-badge&logo=mysql&logoColor=white)|
|하드웨어| <img src="https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white"> ![Arduino](https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=Arduino&logoColor=white)|
|협업| ![js](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white) ![js](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white) ![js](https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white) ![js](https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=Jira&logoColor=white) ![js](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)|

### 🙌 팀원 소개

|이름|담당|
|----|-----|
|이은휘 <br> (팀장)|사과 꽃 개수 체크 및 중복 제거 기능 구현<br> 설계/테스트 정보 문서화<br>사과 나무 시점 별 Aruco 감지 및 거리 추정| 
|이재훈|전체적인 통신 관리 담당 <br> 사과꽃 데이터 라벨링 검수 <br> 장애물 및 사과꽃 모델 학습 <br> 인공수분 데이터 생성|
|김제백|사과꽃 데이터 웹크롤링 및 모델 학습 <br> 장애물 객체 정보에 대한 송수신 패킷 설계 <br> Aruco 갈래 길 예외처리 구현| 
|고선민|GUI 구현, 송수신 패킷 설계 <br> 사과꽃 데이터 웹크롤링 <br> 로봇 보드의 TCP/IP, Serial 통신 테스트 및 모터 제어|
|이시원|차선 인식 및 모터 제어 <br> 장애물 및 사과 꽃 모델 학습 <br> PPT 제작|


<br>

## ⚙️ 시스템 아키텍처

### 1️⃣ 시스템 설계
- 시스템 구성도
- ?
### 2️⃣ 주요 시나리오
- 상황 발생 시나리오 시퀀스 다이어그램
  ?
- 카메라 수동 조작 시나리오 시퀀스 다이어그램 
?
- 상황 종료 시나리오 시퀀스 다이어그램
?

### 3️⃣ 통신 프로토콜 정의
#### Communication Protocol List
|Transmitter|Receiver|Communication Protocol|
|-----|-----|-----|
|CamControlManager (Raspberry Pi)|Main Server|TCP/IP|
|CamControlManager (Raspberry Pi)|PollinationStatus (YOLO model 1)|TCP/IP|
|CamControlManager (Raspberry Pi)|motorControl unit (Arduino Mega)|Serial|
|Main Server|CamControlManager (Raspberry Pi)|TCP/IP|
|Main Server|AppleCare (GUI)|TCP/IP|
|Main Server|ObstacleAvoidance (YOLO model 2)|TCP/IP|
|AppleCare (GUI)|Main Server|TCP/IP|
|PollinationStatus (YOLO model 1)|Main Server|TCP/IP|

#### Command List
<table>
  <thead>
    <tr>
      <th>Transmitter</th>
      <th>Command</th>
      <th>Full Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=2>Main Server</td>
      <td rowspan=1>MC</td>
      <td>Move Control</td>
      <td>motorSpeed 제어. 왼쪽, 오른쪽 모터 값 보내주는 명령어.</td>
    </tr>
    <tr>
      <td rowspan=1>RS</td>
      <td>Robot Status</td>
      <td>로봇에서 전달 받은 카메라 프레임을 읽어 아크로마커 ID 인식을 통해 로봇의 작업 상태 판단. 이를 기반으로 작업 시작, 작업 중, 대기장소 복귀 중, 복귀 완료, 대기 중 (0) 등의 상태로 나누어 로봇에게 전달. 현재는 대기 중 상태 (0)만 전달</td>
    </tr>
    <tr>
      <td rowspan=2>Pollination Status Server</td>
      <td> TS</td>
      <td>Tree Scan</td>
      <td> 나무 스캔 시작(1)과 종료(0) 명령어 전달</td>
    </tr>
    <tr>
      <td> AD </td>
      <td> ArUco Detect</td></td>
      <td> 아르코마커 인식됨(1)과 인식 안됨(0) 명령어 전달</td>
    </tr>
    <tr>
      <td rowspan=1>Cam Control Manager (Robot)</td>
      <td> SF</td>
      <td>Send Frame</td>
      <td> 각각 필요한 서버에 웹캠/광각캠 영상 프레임 전달 (raspi → central, raspi → pollination, central → obstacle)</td>
    </tr>
  <tr>
      <td rowspan=2>AppleCare GUI</td>
      <td rowspan=1>SS</td>
      <td>Set Schedule</td>
      <td>사용자가 지정한 스캔 작업 스케줄 넘버 및 시간 설정값을 전달함.</td>
    </tr>
    <tr>
      <td rowspan=1>RC</td>
      <td>Robot Control</td>
      <td>사용자가 UI상에서 버튼 클릭으로 시작(1), 혹은 강제 대기장소 복귀(0) 명령어 전달. 현재는 시작(1) 버튼만 구현되어있음.</td>
    </tr>
  </tbody>
</table>


#### Packet Structure
<table>
    <thead>
        <tr>
            <th> Command ID </th>
            <th> Data1 </th>
            <th> Data2 </th>
            <th> Data3 </th>
            <th> End byte </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> 2 bytes </td>
            <td>  </td>
            <td>  </td>
            <td>  </td>
            <td> 1 byte </td>
        </tr>
        <tr>
        </tr>
        <tr>
            <td> CS </td>
            <td> 0, 1 (1byte, int)  </td>
            <td> X </td>
            <td> X </td>
            <td> \n </td>
        </tr
        <tr>
            <td> SF </td>
            <td> frame length(4byte)  </td>
            <td> frame </td>
            <td> X </td>
            <td> \n </td>
        </tr>
        <tr>
            <td> MC </td>
            <td> left_speed(1byte, int) </td>
            <td> right_speed(1byte, int) </td>
            <td> X </td>
            <td> \n </td>
        </tr> 
        <tr>
            <td> RC </td>
            <td> 1(1byte, int) </td>
            <td> X</td>
            <td> X </td>
            <td> \n </td>
        </tr>
        <tr>
            <td> RS </td>
            <td> 0(1byte, int) </td>
            <td> X </td>
            <td> X </td>
            <td> \n </td>
        </tr>
        <tr>
            <td> SS </td>
            <td> 0,1 (1byte) </td>
            <td> 1,2,3(1byte, int) </td>
            <td> str (scheduled time/ 5byte) </td>
            <td> \n </td>
        </tr>
        <tr>
            <td> TS </td>
            <td> 0,1 (1byte) </td>
            <td> X </td>
            <td> X </td>
            <td> \n </td>
        </tr>
          <tr>
            <td> AD </td>
            <td> 0,1 (1byte) </td>
            <td> X </td>
            <td> X </td>
            <td> \n </td>
        </tr> 
    </tbody>
</table>


### 4️⃣ GUI 설계
- ?
  
### 5️⃣ Database 설계 
#### 관계정의 개체
- 개화된 꽃, 인공수분된 꽃, 꽃봉오리 수 등 나무의 상태 정보 모니터링
- 나무의 위치정보, 심은날짜, 인공수분 완료 여부
- 장애물 정보
- 현재 작업 단계와 로봇 작업 로그 모니터링
- 로봇 상태 모니터링
- 로봇 대기장소 위치 정보
- 로봇의 작업 스케줄 관리
- ArUco 마커의 위치 및 연결대상 관리
#### 설계시 주안점
1. 중복 배제
   - 나무 모니터링 작업 로그 작성시 Tree 테이블과 TreeCurrentStatus 테이블 생성을 통해 모니터링 데이터를 TreeCurrentStatus에 저장하여 중복하여 데이터가 저장되는것을 배제함 
2. 효율적인 검색
   - TaskLog 테이블의 자료형을 int형으로 생성 및 정규화 작업을 통해 효율적인 검색이 가능하도록 함
3. 동적 할당
   - 로봇, 나무 수의 증가하는 등의 시스템 확장시에도 동적으로 대기장소(station)와 나무를 ArUco 마커가 할당 될 수 있도록 TreeAruco, RobotStationAssignment 테이블을 생성함
  
![DB_schema-Page-1 drawio](https://github.com/user-attachments/assets/298cc87e-9e59-489e-a149-0baf39a65793)

<br>

## 🔗 기능 및 기술 구현

### 1️⃣ 과수원 및 인공수분 모니터링을 위한 도로 주행
#### 주요 기능
- 꽃 상태 체크를 위한 순찰 기능
- 작업 대기 장소 복귀 기능
![Screenshot from 2024-10-04 17-54-44](https://github.com/user-attachments/assets/a2f844c1-23f8-46b2-9682-5b2ee74d7618)


#### 구현 기술
- HSV 색공간에 x, y 방향 경계 활용 및 급회전 구간에 대한 예외 처리 (인식 및 방향)를 통한 ***Line tracking 기술 구현***
- 교차로 (갈래 길)에 대한 방향 설정 및 나무 번호 부여 순서에 따른 로봇 이동을 위한 ***Aruco Marker 기술 구현***

  
### 2️⃣ 사과 나무 꽃 감지 및 인공수분 여부 파악
#### 주요 기능
- 사과 꽃 상태 인식 기능 : 꽃봉우리 혹은 개화한 꽃으로 구분
- 인공수분 완료 여부 체크 기능
![Screenshot from 2024-10-04 17-58-59](https://github.com/user-attachments/assets/05c079cb-0ed0-46a9-af66-c88930a811f0)

#### 구현 기술

### 3️⃣ 도로 위 장애물 감지
#### 주요 기능
#### 구현 기술


### 4️⃣ 인공수분 모니터링
#### 주요 기능
#### 구현 기술





## 시현 영상
- ?
