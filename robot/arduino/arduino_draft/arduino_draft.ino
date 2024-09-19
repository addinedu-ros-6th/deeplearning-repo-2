#define L_MOTOR_IN1 10
#define L_MOTOR_IN2 11
#define L_MOTOR_PWM 12

#define R_MOTOR_IN1 7
#define R_MOTOR_IN2 8
#define R_MOTOR_PWM 13

#define MOTOR_STBY 9  // 모터 스탠바이 핀

int leftSpeed = 0;   // 왼쪽 모터 속도 저장
int rightSpeed = 0;  // 오른쪽 모터 속도 저장

void sendMotorData(int leftSpeed, int rightSpeed)
{
  Serial.print("L:");
  Serial.print(leftSpeed);
  Serial.print(" R:");
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
    if (Serial.available() >= 4) // 총 4바이트 ('M', leftSpeed, rightSpeed, '\n')
    {
      byte packet[4];
      Serial.readBytes(packet, 4); // 패킷 전체를 한 번에 읽음

      if (packet[0] == 'M' && packet[3] == '\n') // 시작 바이트와 종료 바이트 확인
      {
        leftSpeed = packet[1];   // 왼쪽 모터 속도
        rightSpeed = packet[2];  // 오른쪽 모터 속도

        // 모터 제어 로직
        // 왼쪽 모터 제어
        digitalWrite(L_MOTOR_IN1, LOW);
        digitalWrite(L_MOTOR_IN2, HIGH);
        analogWrite(L_MOTOR_PWM, leftSpeed);

        // 오른쪽 모터 제어
        digitalWrite(R_MOTOR_IN1, HIGH);
        digitalWrite(R_MOTOR_IN2, LOW);
        analogWrite(R_MOTOR_PWM, rightSpeed);

        sendMotorData(leftSpeed, rightSpeed);
       }
       else
       {
         // 시작 바이트가 올바르지 않은 경우 패킷 무시
         Serial.println("올바르지 않은 시작 바이트입니다.");
       }
    }
}
