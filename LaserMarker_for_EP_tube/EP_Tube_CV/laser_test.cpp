// 基础 PWM 测试
const int LASER_PIN = 44; // 替换为你连接的 GPIO 引脚
const int PWM_CH = 0;
const int FREQ = 1000;    // 1kHz 频率对于大多数激光模组很稳
const int RES = 8;        // 8位分辨率 (0-255)

void setup() {
  ledcSetup(PWM_CH, FREQ, RES);
  ledcAttachPin(LASER_PIN, PWM_CH);
}

void loop() {
  // 极低功率测试 (例如 5/255)，确认激光能点亮
  ledcWrite(PWM_CH, 5); 
  delay(2000);
  
  // 关闭激光
  ledcWrite(PWM_CH, 0); 
  delay(2000);
}