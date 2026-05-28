#include <Arduino.h>
#include "FastAccelStepper.h"

// --- 引脚定义 ---
#define EN_PIN_X   1  
#define STEP_PIN_X 2  
#define DIR_PIN_X  3  

#define EN_PIN_Y   4  
#define STEP_PIN_Y 5  
#define DIR_PIN_Y  6  

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepperX = NULL;
FastAccelStepper *stepperY = NULL;

const uint32_t MAX_SPEED = 4000;      
const uint32_t ACCELERATION = 15000;  

// 🛑 核心边界设定 (软件限位) 🛑
const int X_MIN = 0;
const int Y_MIN = 0;
// 下面这两个最大值，你需要根据你的物理导轨长度自己试探着改一下
// 比如 16 细分下，3200 步大概是走 40mm (如果是同步带的话)
const int X_MAX = 12000; 
const int Y_MAX = 9000; 

void setup() {
  Serial.begin(115200);
  Serial.println("--- 带有软件限位的坐标中枢已启动 ---");
  Serial.println("⚠️ 警告：请确认开机前已将云台手动推至左下角原点！");
  Serial.println("请输入指令，格式例如: X1600 Y3200");

  engine.init();

  stepperX = engine.stepperConnectToPin(STEP_PIN_X);
  if (stepperX) {
    stepperX->setDirectionPin(DIR_PIN_X);
    stepperX->setEnablePin(EN_PIN_X);
    stepperX->setAutoEnable(true); 
    stepperX->setSpeedInHz(MAX_SPEED);       
    stepperX->setAcceleration(ACCELERATION); 
  }

  stepperY = engine.stepperConnectToPin(STEP_PIN_Y);
  if (stepperY) {
    stepperY->setDirectionPin(DIR_PIN_Y);
    stepperY->setEnablePin(EN_PIN_Y);
    stepperY->setAutoEnable(true);
    stepperY->setSpeedInHz(MAX_SPEED);
    stepperY->setAcceleration(ACCELERATION);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); 

    if (input.length() > 0) {
      int targetX = 0;
      int targetY = 0;

      if (sscanf(input.c_str(), "X%d Y%d", &targetX, &targetY) == 2) {
        
        // 边界保安上线：检查坐标有没有越界
        if (targetX < X_MIN || targetX > X_MAX || targetY < Y_MIN || targetY > Y_MAX) {
          Serial.printf("❌ 越界啦！你输入的坐标 (X:%d, Y:%d) 超出了物理安全边界！\n", targetX, targetY);
          Serial.printf("安全范围是 -> X: [%d, %d], Y: [%d, %d]\n", X_MIN, X_MAX, Y_MIN, Y_MAX);
        } else {
          // 只有通过检查，才允许电机执行
          Serial.printf("🎯 坐标合法，正在飞奔前往 -> X: %d, Y: %d\n", targetX, targetY);
          stepperX->moveTo(targetX);
          stepperY->moveTo(targetY);
        }
        
      } else {
        Serial.println("❌ 格式不对！请严格按照格式，例如: X1600 Y3200 (中间要有空格)");
      }
    }
  }
}