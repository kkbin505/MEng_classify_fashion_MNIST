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
#include <inttypes.h>

// --- TFLite Micro Headers ---
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "..\out_model\model_data_int8.h" // Model file from python
#include "test_image.h"
#include "..\out_model\esp32_test_data.h"


static const char *TAG = "FM_APP";

const int kInputSize = 28*28; 
const int kOutputClasses = 10;
constexpr size_t kTensorArenaSize = 1024 * 1024; 

// -----------------------------------------------------
// 1. GLOBAL OBJECTS 
// -----------------------------------------------------

// Tensor Arena 
uint8_t *tensor_arena = nullptr; 

// TFLite Global Object
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;  
    TfLiteTensor* output = nullptr; 
} 

// Labels for fashion MNIST
const char* kCategoryLabels[kOutputClasses] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};


// -----------------------------------------------------
// 2. FEED INPUT FUNCTION (Quantization)
// -----------------------------------------------------
void feedInput(const uint8_t* source_data) {
    // Model data type check (Int8)
    if (input->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Input type mismatch! Expected Int8, got %d", input->type);
        return;
    }
    
    // Get modal params
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;

    for (int i = 0; i < kInputSize; i++) {
        float val = source_data[i]/255.0f;  // Normalize to [0,1]
        // Quantization: q = val / scale + zero_point
        int32_t q = (int32_t)round(val / scale) + zero_point;
        
        // Constrain int8 range to [-128, 127]
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        
        input->data.int8[i] = (int8_t)q;
    }
}

// -----------------------------------------------------
// 3. APP_MAIN
// -----------------------------------------------------
extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting Fashion MNIST (Full INT8 Mode)...");

    // 1. Use SPI memary （2MB）
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate arena in PSRAM!");
        while(1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    ESP_LOGI(TAG, "Arena allocated.");

    // 2. Load model
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch!"); while(1) vTaskDelay(1000);
    }

    // 3. Resolver
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D(); 
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize(); 
    resolver.AddDequantize(); 
    resolver.AddShape();
    resolver.AddStridedSlice(); 
    resolver.AddPack();
    resolver.AddPad(); 

    // 4. Interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!"); while (1) vTaskDelay(1000);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Entering Loop...");

    while (1) {
        //Quantization model to int8
        // feedInput(g_test_input_data);

        // // 2. prediction
        // if (interpreter->Invoke() != kTfLiteOk) {
        //     ESP_LOGE(TAG, "Invoke failed!");
        // } else {
        //     // 3. 反量化输出
        //     float scale = output->params.scale;
        //     int zero_point = output->params.zero_point;
        //     int predicted = -1;
        //     float max_prob = -1.0f;

        //     for (int i = 0; i < kOutputClasses; i++) {
        //         int8_t q_val = output->data.int8[i];
        //         float prob = (q_val - zero_point) * scale;
        //         if (prob > max_prob) {
        //             max_prob = prob;
        //             predicted = i;
        //         }
        //     }
        //     ESP_LOGI(TAG, "Prediction: %s (%.2f%%)", kCategoryLabels[predicted], (double)(max_prob * 100.0f));
        // }
        // vTaskDelay(pdMS_TO_TICKS(2000));
    for (int img_idx = 0; img_idx < 30; img_idx++) {
        // Feed one test image
        feedInput(test_images[img_idx]);

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
            continue;
        }

        // Dequantize output
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        int predicted = -1;
        float max_prob = -1.0f;

        for (int i = 0; i < kOutputClasses; i++) {
            int8_t q_val = output->data.int8[i];
            float prob = (q_val - zero_point) * scale;
            if (prob > max_prob) {
                max_prob = prob;
                predicted = i;
            }
        }

        ESP_LOGI(TAG, "Image %d prediction: %s (%.2f%%)", img_idx, kCategoryLabels[predicted], (double)(max_prob*100));
        vTaskDelay(pdMS_TO_TICKS(500)); // optional delay
    }

        }
}