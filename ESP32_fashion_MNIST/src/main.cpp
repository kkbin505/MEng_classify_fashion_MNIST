#include <Arduino.h>
#include <esp_heap_caps.h> 

// --- TFLite Micro Core Headers ---
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Use Mutable resolver
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"    // <-- REQUIRED
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data_int16.h" 

// =========== Step 1: 内存分配和全局变量 ==================
constexpr size_t kTensorArenaSize = 512 * 1024; // 512 KB
uint8_t *tensor_arena = nullptr; // Initialize as nullptr

// --- 全局 TFLite 对象 ---
namespace {
tflite::ErrorReporter* error_reporter = nullptr; // <<< FIX 1
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
} 

// --- Model Configuration ---
const int kInputWidth = 28;
const int kInputHeight = 28;
const int kInputSize = kInputWidth * kInputHeight;
const int kOutputClasses = 10;

// --- Labels (Example) ---
const char* kCategoryLabels[kOutputClasses] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

// --- Test Input Data (Float 0~1) ---
// Note: This test pattern must be replaced with your actual image data!
float g_dummy_input_data[kInputSize];


// Quantization data from float 32 to int 16
void feedInput(const float* source_data) {
    // 检查输入类型是否为 Int16
    if (input->type != kTfLiteInt16) {
        Serial.println("FATAL: Input tensor type is not Int16!");
        while(1);
    }
    
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;

    for (int i = 0; i < kInputSize; i++) {
        float val = source_data[i]; // 0.0 to 1.0 float value
        
        // **量化公式:** Q = round(Value / Scale) + ZeroPoint
        // 使用 round() 函数进行四舍五入，提高精度
        int32_t q = (int32_t)round(val / scale) + zero_point;
        
        // 裁剪：确保结果在 int16 的有效范围 [-32768, 32767] 内
        q = max(-32768, min(32767, q));
        
        // 写入 int16 内存
        input->data.i16[i] = (int16_t)q; 
    }
}

// =========== Step 3: 初始化 TensorFlow Lite =================
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n[ESP32-S3 TFLite Test] Starting...");

    // 1. 初始化错误报告器 (FIX 1)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // 2. PSRAM 分配 (使用更稳定的 heap_caps_malloc)
    if (!psramFound()) {
        Serial.println("❌ PSRAM not found!"); while (1);
    }
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
    if (!tensor_arena) {
        Serial.println("❌ Failed to allocate tensor_arena in PSRAM!"); while (1);
    }
    Serial.printf("✅ Allocated %d bytes tensor_arena in PSRAM\n", kTensorArenaSize);

    // 3. 加载模型
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ Model schema version mismatch!"); while (1);
    }

    // 4. 定义操作解析器 (FIX 3: 使用 Mutable Resolver)
    static tflite::MicroMutableOpResolver<6> resolver; // Assuming 5 core ops: FC, Softmax, ReLU
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    // --- 关键修复：添加 SHAPE 和 RESHAPE ---
    resolver.AddShape();          // <<< ADD THIS LINE (Fixes the current error)
    resolver.AddReshape();        // <<< ADD THIS LINE (Often implicitly used by Flatten)
    resolver.AddStridedSlice();   // <<< ADD THIS LINE

    // 5. 创建解释器 (FIX 1: 传入 error_reporter)
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter); 
    interpreter = &static_interpreter;

    // 6. 分配张量
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("❌ AllocateTensors() failed");
        // 如果这里失败，需要增加 kTensorArenaSize
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("✅ TFLite initialized successfully.");

    // Initialize dummy input data (using a simple pattern for test)
    for (int i = 0; i < kInputSize; i++) {
        // Creates a gradient pattern (0.0 to 1.0) for basic input test
        g_dummy_input_data[i] = (float)(i % 28) / 28.0f; 
    }

}

// =========== Step 4: 运行推理 ===============================
void loop() {
    // 构造一个测试输入（28x28 灰度图, FP32 I/O model assumes data.f)

    // 1. Fill Input Tensor with Quantized Data
    feedInput(g_dummy_input_data); // <<< Use the correct input filling function

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ Inference failed!");
        return;
    }

    // 1. 获取反量化参数
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    int predicted_class = -1;
    float max_prob = -1.0f; // Initialize with a low float value

    // 2. 循环读取并反量化结果
    for (int i = 0; i < kOutputClasses; i++) {
        // **反量化公式:** Float = (Quantized_Value - ZeroPoint) * Scale
        int16_t quantized_value = output->data.i16[i]; // Read Int16 value
        
        float prob = (quantized_value - zero_point) * scale;
        
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

// 5. Print Result
    Serial.println("\n--- Inference Result ---");
    Serial.printf("Predicted Class: %s\n", kCategoryLabels[predicted_class]);
    Serial.printf("Confidence: %.2f%%\n", max_prob * 100.0f);
    Serial.println("------------------------");

    delay(3000);
}