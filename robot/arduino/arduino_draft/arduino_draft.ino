#define L_MOTOR_IN1 10
#define L_MOTOR_IN2 11
#define L_MOTOR_PWM 12

#define R_MOTOR_IN1 7
#define R_MOTOR_IN2 8
#define R_MOTOR_PWM 13

#define L_LED 44  // 왼쪽 LED 핀
#define R_LED 6   // 오른쪽 LED 핀
 

#define MOTOR_STBY 9  // 모터 스탠바이 핀


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

    pinMode(L_LED, OUTPUT);
    pinMode(R_LED, OUTPUT);
    digitalWrite(L_LED, LOW);  // 왼쪽 LED 기본 상태: OFF
    digitalWrite(R_LED, LOW);  // 오른쪽 LED 기본 상태: OFF

    Serial.begin(115200);              // 시리얼 통신 시작
}

void loop()
{
    if (Serial.available() > 0) 
    {
      String input = Serial.readStringUntil('\n');  // 시리얼 입력 읽기
      String command = input.substring(0, 1);  // 첫 번째 문자는 명령어
      // char command = input.charAt(0);
      int separatorIndex = input.indexOf(',');  // 쉼표 위치 찾기
      int leftSpeed = input.substring(1, separatorIndex).toInt();  // 왼쪽 속도 값 추출
      int rightSpeed = input.substring(separatorIndex + 1).toInt();  // 오른쪽 속도 값 추출
      
      digitalWrite(L_LED, HIGH);
      digitalWrite(R_LED, HIGH);
      if (command == "M") // 시작 바이트와 종료 바이트 확인
      {
        // 모터 제어 로직
        // leftSpeed = input.substring(1, separatorIndex).toInt();
        // rightSpeed = input.substring(separatorIndex + 1).toInt();
            digitalWrite(L_MOTOR_IN1, LOW);
            digitalWrite(L_MOTOR_IN2, HIGH);
            analogWrite(L_MOTOR_PWM, leftSpeed);

            digitalWrite(R_MOTOR_IN1, HIGH);
            digitalWrite(R_MOTOR_IN2, LOW);
            analogWrite(R_MOTOR_PWM, rightSpeed);

            //digitalWrite(L_LED, HIGH);  // 왼쪽 LED 켜기
            //digitalWrite(R_LED, HIGH);  // 오른쪽 LED 켜기
            //delay(1000);
        // sendMotorData(leftSpeed, rightSpeed);
       }
   }

    // else
    // {
    //     analogWrite(L_MOTOR_PWM, 0);
    //     analogWrite(R_MOTOR_PWM, 0);

    //     digitalWrite(L_LED, LOW);
    //     digitalWrite(R_LED, LOW);
    // }

    //else
    //{
    //   analogWrite(L_MOTOR_PWM, 0);
    //   analogWrite(R_MOTOR_PWM, 0);
    //}
}
