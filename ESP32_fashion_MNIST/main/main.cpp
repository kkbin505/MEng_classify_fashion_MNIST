/*
 * Fashion MNIST Classification on ESP32-S3
 * Compatible with Hybrid Models (Float32 Input/Output)
 */

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string.h> // For memcpy
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h" 
#include "esp_system.h"

// --- TFLite Micro Headers ---
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data_int16.h" // 你的模型文件
// #include "test_image.h"
// Boot
const float g_test_input_data[784] = {
0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0039f, 0.0000f, 0.0000f, 0.0667f, 0.0000f, 0.1373f, 0.2157f, 0.2039f, 0.1765f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0118f, 0.0000f, 0.0039f, 0.9804f, 1.0000f, 0.9608f, 0.9961f, 0.9333f, 0.9569f, 0.9373f, 0.5412f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3451f, 0.4863f, 0.6667f, 0.9961f, 0.5412f, 0.7333f, 1.0000f, 0.7333f, 0.1255f, 0.0157f, 0.0000f, 0.0039f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3294f, 0.3843f, 0.0000f, 0.7137f, 0.8235f, 0.9529f, 1.0000f, 0.1451f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0039f, 0.0000f, 0.0235f, 0.2196f, 0.2824f, 0.3569f, 0.5216f, 0.1686f, 0.0000f, 0.9412f, 0.8549f, 0.0000f, 0.0000f, 0.1529f, 0.1804f, 0.0784f, 0.0980f, 0.0078f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1882f, 0.4275f, 0.2745f, 0.2118f, 0.1725f, 0.2784f, 0.2196f, 0.2510f, 0.0588f, 0.0784f, 0.1137f, 0.1098f, 0.2275f, 0.2235f, 0.2000f, 0.0784f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2549f, 0.2863f, 0.3216f, 0.1922f, 0.2275f, 0.2039f, 0.1255f, 0.3294f, 0.2706f, 0.0980f, 0.1961f, 0.2471f, 0.1804f, 0.1059f, 0.0980f, 0.1137f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0471f, 0.3059f, 0.2078f, 0.5137f, 0.1451f, 0.2235f, 0.2039f, 0.0784f, 0.3529f, 0.3059f, 0.0784f, 0.2078f, 0.2431f, 0.1412f, 0.0667f, 0.1059f, 0.1529f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1098f, 0.3333f, 0.1137f, 0.6039f, 0.2275f, 0.1843f, 0.1686f, 0.0471f, 0.2980f, 0.2784f, 0.0824f, 0.1333f, 0.0745f, 0.0824f, 0.0745f, 0.1294f, 0.1686f, 0.0275f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1686f, 0.3098f, 0.0510f, 0.5373f, 0.2549f, 0.1608f, 0.1647f, 0.0392f, 0.3294f, 0.2627f, 0.0510f, 0.1137f, 0.1098f, 0.0706f, 0.1608f, 0.1765f, 0.1137f, 0.0824f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2078f, 0.2863f, 0.0392f, 0.5725f, 0.3333f, 0.1686f, 0.1647f, 0.0353f, 0.3294f, 0.2471f, 0.0627f, 0.1216f, 0.0941f, 0.0549f, 0.1294f, 0.1137f, 0.1137f, 0.0510f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2392f, 0.2745f, 0.0078f, 0.6627f, 0.4000f, 0.1098f, 0.1843f, 0.0588f, 0.3137f, 0.2353f, 0.0392f, 0.1137f, 0.1020f, 0.0000f, 0.3020f, 0.1098f, 0.1059f, 0.0549f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2627f, 0.2353f, 0.0118f, 0.7176f, 0.3059f, 0.1725f, 0.1804f, 0.0510f, 0.2941f, 0.2431f, 0.0353f, 0.0941f, 0.1098f, 0.0000f, 0.6314f, 0.1529f, 0.0510f, 0.0824f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2902f, 0.1961f, 0.0157f, 0.8706f, 0.2745f, 0.1451f, 0.1804f, 0.0627f, 0.2941f, 0.2549f, 0.0275f, 0.1020f, 0.0627f, 0.0000f, 0.9490f, 0.1804f, 0.0275f, 0.0980f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2863f, 0.1412f, 0.0431f, 1.0000f, 0.2235f, 0.1725f, 0.2118f, 0.0431f, 0.2902f, 0.2471f, 0.0157f, 0.1020f, 0.0235f, 0.0039f, 0.8549f, 0.2863f, 0.0000f, 0.1059f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0314f, 0.2941f, 0.1137f, 0.0980f, 1.0000f, 0.2627f, 0.1804f, 0.1961f, 0.0235f, 0.3098f, 0.2471f, 0.0314f, 0.0980f, 0.0000f, 0.1059f, 0.9569f, 0.3961f, 0.0000f, 0.1137f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0431f, 0.2941f, 0.0588f, 0.2078f, 1.0000f, 0.2275f, 0.1529f, 0.1922f, 0.0706f, 0.2980f, 0.2549f, 0.0235f, 0.1059f, 0.0157f, 0.0000f, 0.8627f, 0.5412f, 0.0000f, 0.1098f, 0.0118f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0431f, 0.2902f, 0.0196f, 0.4039f, 0.9961f, 0.1922f, 0.1882f, 0.1804f, 0.0510f, 0.2863f, 0.2549f, 0.0078f, 0.0980f, 0.0196f, 0.0000f, 0.8196f, 0.6941f, 0.0000f, 0.1176f, 0.0275f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0627f, 0.2941f, 0.0157f, 0.4745f, 1.0000f, 0.1412f, 0.1843f, 0.2039f, 0.0627f, 0.2627f, 0.2706f, 0.0078f, 0.0863f, 0.0549f, 0.0000f, 0.7451f, 0.7098f, 0.0000f, 0.1098f, 0.0314f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0863f, 0.3059f, 0.0000f, 0.5098f, 0.9961f, 0.0824f, 0.2314f, 0.2275f, 0.1098f, 0.2902f, 0.2824f, 0.0039f, 0.1059f, 0.0941f, 0.0000f, 0.6863f, 0.8000f, 0.0000f, 0.0941f, 0.0392f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0902f, 0.3020f, 0.0000f, 0.6078f, 0.8549f, 0.0784f, 0.2235f, 0.2078f, 0.0941f, 0.2745f, 0.2863f, 0.0078f, 0.1059f, 0.0863f, 0.0000f, 0.5255f, 0.8392f, 0.0000f, 0.0784f, 0.0471f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0941f, 0.2941f, 0.0000f, 0.7255f, 0.7451f, 0.0824f, 0.2510f, 0.2314f, 0.1294f, 0.2824f, 0.2824f, 0.0157f, 0.1020f, 0.1216f, 0.0000f, 0.4784f, 0.8549f, 0.0118f, 0.0667f, 0.0627f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0941f, 0.3176f, 0.0000f, 0.7608f, 0.6157f, 0.0706f, 0.2235f, 0.2196f, 0.1176f, 0.2784f, 0.3020f, 0.0157f, 0.0902f, 0.1020f, 0.0000f, 0.4353f, 0.8902f, 0.0510f, 0.0510f, 0.0745f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1255f, 0.3059f, 0.0000f, 0.8863f, 0.5333f, 0.2000f, 0.3216f, 0.2863f, 0.1529f, 0.2941f, 0.3137f, 0.0314f, 0.1098f, 0.1294f, 0.0000f, 0.4000f, 0.9490f, 0.0627f, 0.0471f, 0.0745f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1412f, 0.2745f, 0.0118f, 0.9176f, 0.3294f, 0.2039f, 0.2941f, 0.2941f, 0.2235f, 0.2510f, 0.2588f, 0.0745f, 0.1608f, 0.1529f, 0.0314f, 0.2039f, 0.8549f, 0.1765f, 0.0235f, 0.0667f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1373f, 0.2706f, 0.1137f, 0.9412f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.8980f, 0.4353f, 0.0000f, 0.0667f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2588f, 0.3294f, 0.1765f, 0.4510f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.4667f, 0.3059f, 0.0941f, 0.1020f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2118f, 0.2784f, 0.1216f, 0.2000f, 0.0000f, 0.0000f, 0.0039f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1412f, 0.1176f, 0.1059f, 0.1059f, 0.0000f, 0.0000f, 0.0000f, 0.0000f
};



static const char *TAG = "FM_APP";

const int kInputSize = 784; 
const int kOutputClasses = 10;
constexpr size_t kTensorArenaSize = 512 * 1024; 
uint8_t *tensor_arena = nullptr; 

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
} 

// Test Input Data (Simulated T-shirt/Top)
// float g_test_input_data[kInputSize];

const char* kCategoryLabels[kOutputClasses] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting Fashion MNIST (Float32 I/O Mode)...");

    // 1. PSRAM Allocation
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
    if (tensor_arena == nullptr) { 
        ESP_LOGE(TAG, "FATAL: Failed to allocate arena!"); while (1) vTaskDelay(1000); 
    }

    // 2. Load Model
    model = tflite::GetModel(g_model_data); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!"); while(1) vTaskDelay(1000);
    }

    // 3. Resolver
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize(); 
    resolver.AddDequantize(); 
    resolver.AddShape();
    resolver.AddStridedSlice(); 
    resolver.AddPack();

    // 4. Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // 5. Allocate
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!"); while (1) vTaskDelay(1000);
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // --- 关键检查：确认是 Float32 ---
    if (input->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Unexpected Input Type: %d (Expected 1=Float32)", input->type);
    } else {
        ESP_LOGI(TAG, "Input Type Verified: Float32");
    }

    // 填充测试数据 (简单的渐变，或者是你之前准备的靴子数据)
    // 你的 T-shirt/Top 数据:
    // for (int i = 0; i < kInputSize; i++) {
    //    // 这里用简单的模式，如果你有真实数据数组，用 memcpy
    //    g_test_input_data[i] = (float)(i % 28) / 28.0f; 
    // }

    ESP_LOGI(TAG, "Entering Inference Loop...");
    
    while (1) {
        // 1. 直接复制 Float 数据 (不需要 feedInput 函数)
        memcpy(input->data.f, g_test_input_data, kInputSize * sizeof(float));

        // 2. 推理
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
        } else {
            // 3. 直接读取 Float 输出 (不需要反量化)
            int predicted_class = -1;
            float max_prob = -1.0f; 

            for (int i = 0; i < kOutputClasses; i++) {
                // 直接读取 float
                float prob = output->data.f[i]; 
                if (prob > max_prob) {
                    max_prob = prob;
                    predicted_class = i;
                }
            }

            ESP_LOGI(TAG, "Prediction: %s (%.2f%%)", kCategoryLabels[predicted_class], (double)(max_prob * 100.0f));
        }

        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}