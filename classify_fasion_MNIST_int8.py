# train_and_export_int8_fashion_mnist.py
# Requirements: python3, tensorflow>=2.6, numpy
# Recommended: Use Conda environment on Windows and install TensorFlow
# Run: python train_and_export_int8_fashion_mnist.py

import tensorflow as tf
import numpy as np
import os

# ------------ Settings ------------
MODEL_DIR = "out_model"
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 12
BATCH_SIZE = 128
REPRESENTATIVE_SAMPLES = 500  # Representative samples for PTQ
# ----------------------------------

# 1) Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32)  / 255.0

# Reshape -> [N,28,28,1]
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# 2) Build model (lightweight CNN)
def build_model():
    inp = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)  # no softmax -> from_logits=True
    return tf.keras.Model(inputs=inp, outputs=out)

model = build_model()
model.summary()

# 3) Training
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Data augmentation (optional)
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.08
)
train_gen = data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE)

model.fit(
    train_gen,
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

# 4) Save Keras model
keras_model_path = os.path.join(MODEL_DIR, "model_fp32.keras")
model.save(keras_model_path)
print("Keras model saved to:", keras_model_path)

# 5) Representative dataset generator (for PTQ)
def representative_dataset_gen():
    indices = np.random.choice(len(x_train), REPRESENTATIVE_SAMPLES, replace=False)
    for i in indices:
        img = x_train[i:i+1].astype(np.float32)
        yield [img]

# 6) Convert to TFLite int8 (full integer quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, "model_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Wrote TFLite int8 model to:", tflite_path)

# 7) Optional: evaluate quantized model on CPU
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
i_idx = interpreter.get_input_details()[0]['index']
i_info = interpreter.get_input_details()[0]
o_idx = interpreter.get_output_details()[0]['index']
o_info = interpreter.get_output_details()[0]

in_scale, in_zero_point = i_info['quantization']
out_scale, out_zero_point = o_info['quantization']

correct = 0
N_eval = 2000
for i in range(N_eval):
    img = x_test[i:i+1]
    img_q = np.clip(np.round(img / in_scale + in_zero_point), -128, 127).astype(np.int8)
    interpreter.set_tensor(i_idx, img_q)
    interpreter.invoke()
    out_q = interpreter.get_tensor(o_idx)
    out_f = (out_q.astype(np.float32) - out_zero_point) * out_scale
    pred = np.argmax(out_f, axis=-1)[0]
    if pred == y_test[i]:
        correct += 1
print(f"Quantized model accuracy on {N_eval} samples: {correct/N_eval:.4f}")

# 8) Export .tflite as C header for TFLite Micro
def write_c_array(tflite_bytes, h_path, var_name="g_model_data"):
    with open(h_path, "w") as f:
        f.write("// Automatically generated from model_int8.tflite\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}")
            if i != len(tflite_bytes)-1:
                f.write(", ")
            if (i+1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print("C header written to:", h_path)

with open(tflite_path, "rb") as f:
    tflite_bytes = f.read()
write_c_array(tflite_bytes, os.path.join(MODEL_DIR, "model_data_int8.h"))

print("All done. Check folder:", MODEL_DIR)
