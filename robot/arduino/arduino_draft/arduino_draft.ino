#define L_MOTOR_IN1 10
#define L_MOTOR_IN2 11
#define L_MOTOR_PWM 12

#define R_MOTOR_IN1 7
#define R_MOTOR_IN2 8
#define R_MOTOR_PWM 13

#define MOTOR_STBY 9  // 모터 스탠바이 핀

void setup() {
    pinMode(L_MOTOR_IN1, OUTPUT);
    pinMode(L_MOTOR_IN2, OUTPUT);
    pinMode(L_MOTOR_PWM, OUTPUT);

    pinMode(R_MOTOR_IN1, OUTPUT);
    pinMode(R_MOTOR_IN2, OUTPUT);
    pinMode(R_MOTOR_PWM, OUTPUT);

    pinMode(MOTOR_STBY, OUTPUT);
    digitalWrite(MOTOR_STBY, HIGH);  // 모터 스탠바이 해제
    Serial.begin(9600);              // 시리얼 통신 시작
}

void loop() 
{
  if (Serial.available()) 
  {
    String packet = Serial.readStringUntil('\n');  // 개행 문자('\n')가 올 때까지 읽기
    
    if (packet.length() >= 3) // 최소 3바이트 이상이어야 유효한 패킷
    {
      int command = packet[0];  // 1바이트 명령 ID 수신 (10)
      int leftSpeed = packet[1];  // 1바이트 왼쪽 속도 수신 (0~255)
      int rightSpeed = packet[2];  // 1바이트 오른쪽 속도 수신 (0~255)

      if (command == 10) 
      {  
        // 명령 ID가 10인지 확인 후 모터 제어
        digitalWrite(L_MOTOR_IN1, LOW);
        digitalWrite(L_MOTOR_IN2, HIGH);
        analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

        digitalWrite(R_MOTOR_IN1, HIGH);
        digitalWrite(R_MOTOR_IN2, LOW);
        analogWrite(R_MOTOR_PWM, rightSpeed);  // 오른쪽 모터 설정
      } 
    }
  }
}
