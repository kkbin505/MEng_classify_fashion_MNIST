#include <stdio.h>       // For printf
#include <stdlib.h>      // For malloc, free
#include <math.h>        // For round()
#include <algorithm>     // For std::min/max
#include "string.h"      // For memcpy

// --- ESP-IDF and FreeRTOS Headers ---
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"     // For logging (ESP_LOGI)
#include "esp_system.h" 
#include "esp_heap_caps.h" 
#include "esp_spi_flash.h"
#include "esp_psram.h"   // For PSRAM check

// --- TFLite Micro Core Headers ---
#include "micro_mutable_op_resolver.h" 
#include "micro_interpreter.h"
#include "micro_error_reporter.h"    
#include "schema_generated.h"
#include "version.h"

#include "model_data_int16.h" 

// 定义日志 TAG
static const char *TAG = "TFLITE_APP";

// (Your Global Variables and Definitions Here...)

// ... (Your feedInput function) ...

// -----------------------------------------------------
// 2. MAIN APPLICATION ENTRY POINT (app_main)
// -----------------------------------------------------
extern "C" void app_main() {
    // 启动串口 (使用标准的 printf 即可，ESP-IDF 会处理)
    ESP_LOGI(TAG, "Starting TFLite Application...");

    // 1. 初始化错误报告器
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // 2. PSRAM 检查和分配
    // 使用 ESP-IDF 官方函数检查 PSRAM
    if (esp_spiram_get_size() == 0) { 
        ESP_LOGE(TAG, "❌ PSRAM NOT DETECTED (Size 0). Halting."); 
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
    // 分配 Tensor Arena
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); 
    if (!tensor_arena) { 
        ESP_LOGE(TAG, "❌ Failed to allocate %d bytes for arena!", kTensorArenaSize); 
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    ESP_LOGI(TAG, "✅ Allocated %d bytes tensor_arena in PSRAM.", kTensorArenaSize);

    // 3. Load Model and Resolver
    // ... (Your existing model loading and resolver code) ...
    // Note: Replace all Serial.printf/println with ESP_LOGI or printf

    // 4. Create Interpreter and Allocate Tensors
    // ... (Your existing interpreter creation and allocation logic) ...
    
    // 5. Inference Loop
    while (1) {
        // ... (Your feedInput, interpreter->Invoke(), and result parsing logic) ...
        
        // 替换 delay()
        vTaskDelay(pdMS_TO_TICKS(3000)); 
    }
}