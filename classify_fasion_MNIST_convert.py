import tensorflow as tf
import numpy as np
import os

# --- 配置参数 ---
TFLITE_FILE_NAME = 'fashion_mnist_quant_int16.tflite'
C_HEADER_NAME = 'model_data_int16.h'
# 假设您的模型已在内存中或可以从文件加载
# 请替换为您实际加载训练数据的代码
TRAIN_DATA_SAMPLES = 100 

print("TensorFlow Version:", tf.__version__)

# ====================================================================
# --- 1. 定义数据加载和模型 (根据您的实际环境修改) ---
# ====================================================================

# 示例: 加载 Fashion MNIST 数据 (用于校准)
try:
    (train_images_raw, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images_raw.astype(np.float32) / 255.0
    
    # 示例: 加载或重建您的模型结构
    # 确保这里的模型结构与您训练时使用的完全一致
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # 如果您的模型已保存，请使用 model = tf.keras.models.load_model('your_model.h5')
    # 为了让脚本运行，我们假设模型已经被编译和训练（或者至少是加载的）
    
except Exception as e:
    print(f"FATAL: Could not load data or define model. Error: {e}")
    exit()

# ====================================================================
# --- 2. TFLite 转换函数 ---
# ====================================================================

def convert_to_full_int16_c_array(keras_model, training_data):
    
    # 1. 定义校准数据集生成器
    def representative_data_gen():
        # 仅使用前 X 个样本进行校准
        for input_value in tf.data.Dataset.from_tensor_slices(training_data).batch(1).take(TRAIN_DATA_SAMPLES):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 设置校准数据集
    converter.representative_dataset = representative_data_gen

    # 强制 FULL INT16 接口和内部计算 (解决 ESP32 兼容性问题)
    converter.target_spec.supported_types = [tf.int16] 
    # converter.inference_input_type = tf.int16
    # converter.inference_output_type = tf.int16

    # 执行转换
    tflite_model_binary = converter.convert()
    return tflite_model_binary

# ====================================================================
# --- 3. 执行转换并生成 C 数组 ---
# ====================================================================

print("Starting Full INT16 Quantization...")
tflite_model_binary = convert_to_full_int16_c_array(model, train_images)
print(f"Conversion successful. Model size: {len(tflite_model_binary) / 1024:.2f} KB")

# 1. 保存 TFLite 文件（可选，用于调试）
with open(TFLITE_FILE_NAME, 'wb') as f:
    f.write(tflite_model_binary)

# 2. 将 TFLite 文件内容转换为 C 数组
c_code = (
    '#ifndef MODEL_DATA_H\n'
    '#define MODEL_DATA_H\n\n'
    '// Full INT16 Quantized Model Data (Size: %.2f KB)\n' % (len(tflite_model_binary) / 1024.0) +
    'const unsigned char g_model_data[] = {\n  ' + 
    # 将每个字节转换为十六进制字符串 (0xAB)
    ', '.join(f'0x{b:02x}' for b in tflite_model_binary) + 
    '\n};\n'
    f'const int g_model_data_len = {len(tflite_model_binary)};\n\n'
    '#endif // MODEL_DATA_H\n'
)

# 3. 写入 C 头文件
with open(C_HEADER_NAME, 'w') as f:
    f.write(c_code)

print(f"\n🎉 Deployment Header Created! Replace include/{C_HEADER_NAME} in your PlatformIO project.")