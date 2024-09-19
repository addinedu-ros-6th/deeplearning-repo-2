#define L_MOTOR_IN1 10
#define L_MOTOR_IN2 11
#define L_MOTOR_PWM 12

#define R_MOTOR_IN1 7
#define R_MOTOR_IN2 8
#define R_MOTOR_PWM 13

#define MOTOR_STBY 9  // 모터 스탠바이 핀

int leftSpeed = 0;  // 왼쪽 모터 속도 저장
int rightSpeed = 0;  // 오른쪽 모터 속도 저장

void sendMotorData(int leftSpeed, int rightSpeed)
{
  Serial.print("L:");
  Serial.print(leftSpeed);
  Serial.print("R:");
  Serial.print(rightSpeed);
  Serial.print("\n");
}

void setup() 
{
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
      leftSpeed = packet[0];  // 1바이트 왼쪽 속도 수신 (0~255)
      rightSpeed = packet[1];  // 1바이트 오른쪽 속도 수신 (0~255)

       
      // 명령 ID가 10인지 확인 후 모터 제어
      digitalWrite(L_MOTOR_IN1, LOW);
      digitalWrite(L_MOTOR_IN2, HIGH);
      analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

      digitalWrite(R_MOTOR_IN1, HIGH);
      digitalWrite(R_MOTOR_IN2, LOW);
      analogWrite(R_MOTOR_PWM, rightSpeed);  // 오른쪽 모터 설정

      sendMotorData(leftSpeed, rightSpeed);      
    }
  }
}
