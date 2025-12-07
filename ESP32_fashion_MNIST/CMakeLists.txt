cmake_minimum_required(VERSION 3.16.0)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ESP32_fashion_MNIST)
# CMakeLists.txt for the TFLite Micro component

# This explicitly adds the necessary internal directories to the include path.
idf_component_register(
    # Source files are typically handled by internal CMake logic
    SRCS "" 
    
    # Expose key header directories
    INCLUDE_DIRS 
        "tensorflow/lite/micro"
        "tensorflow/lite/micro/kernels"
        "tensorflow/lite/core/api"
    
)