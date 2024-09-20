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

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // 시리얼 입력 읽기
    String command = input.substring(0, 2);  // 첫 번째 문자는 명령어
    int separatorIndex = input.indexOf(',');  // 쉼표 위치 찾기
    int leftSpeed = input.substring(2, separatorIndex).toInt();  // 왼쪽 속도 값 추출
    int rightSpeed = input.substring(separatorIndex + 1).toInt();  // 오른쪽 속도 값 추출
    
    // leftSpeed = 40
    // rightSpeed = 47
    
    if (command == "MF") {
      // 모터 전진
      digitalWrite(L_MOTOR_IN1, LOW);
      digitalWrite(L_MOTOR_IN2, HIGH);
      analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

      digitalWrite(R_MOTOR_IN1, HIGH);
      digitalWrite(R_MOTOR_IN2, LOW);
      analogWrite(R_MOTOR_PWM, rightSpeed);  // 오른쪽 모터 설정
    } 
    else if (command == "MB") {
      // 모터 후진
      digitalWrite(L_MOTOR_IN1, HIGH);
      digitalWrite(L_MOTOR_IN2, LOW);
      analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

      digitalWrite(R_MOTOR_IN1, LOW);
      digitalWrite(R_MOTOR_IN2, HIGH);
      analogWrite(R_MOTOR_PWM, rightSpeed);  // 오른쪽 모터 설정
    } 
    else if (command == "ML") {
      // 좌회전
      digitalWrite(L_MOTOR_IN1, HIGH);
      digitalWrite(L_MOTOR_IN2, LOW);
      analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

      digitalWrite(R_MOTOR_IN1, HIGH);
      digitalWrite(R_MOTOR_IN2, LOW);
      analogWrite(R_MOTOR_PWM, rightSpeed);
    } 
    else if (command == "MR") {
      // 우회전
      digitalWrite(L_MOTOR_IN1, LOW);
      digitalWrite(L_MOTOR_IN2, HIGH);
      analogWrite(L_MOTOR_PWM, leftSpeed);  // 왼쪽 모터 설정

      digitalWrite(R_MOTOR_IN1, LOW);
      digitalWrite(R_MOTOR_IN2, HIGH);
      analogWrite(R_MOTOR_PWM, rightSpeed);  // 오른쪽 모터 설정
    } 
    else if (command == "MS") {
      // 모터 정지
      analogWrite(L_MOTOR_PWM, 0);
      analogWrite(R_MOTOR_PWM, 0);
    }
  }
}
