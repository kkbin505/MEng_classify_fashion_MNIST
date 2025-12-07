import numpy as np
from PIL import Image
import os
import tensorflow as tf

# --- Configuration ---
IMAGE_PATH = 'img/test_tshirt.jpg' # <-- Replace with your target image path
TARGET_SIZE = 28
OUTPUT_SHAPE = (1, TARGET_SIZE, TARGET_SIZE, 1)

# 1. Load the saved model
model = tf.keras.models.load_model("fashion_mnist_model.keras")
print("Model loaded successfully!")

def load_image(image_path):
    # 1. Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    # 2. Resize the image
    
    img = img.resize((28, 28))
    # 3. Convert to numpy array and normalize to [0, 1]
    img_visual = np.array(img, dtype=np.uint8) 
    img_array = np.array(img, dtype=np.float32) / 255.0
    # 4. Add batch dimension
    img_array = img_array.reshape(1, 28, 28)
    return img_array , img_visual

# Load and preprocess the image
new_image,img_visual = load_image(IMAGE_PATH)

print(new_image.shape)

if new_image is not None:
        
        # --- 1. Save NumPy array (Precise input for the model) ---
        np_save_path = 'preprocessed_input.npy'
        np.save(np_save_path, new_image)
        print(f"✅ Saved NumPy array to: {np_save_path}")

        # --- 2. Save JPG image (For visual verification) ---
        # Image data must be in the 0-255 range for saving
        img_to_save = Image.fromarray(img_visual)
        jpg_save_path = 'img/preprocessed_28x28.jpg'
        
        # Note: Ensure grayscale mode ('L') for saving as JPG/PNG
        if img_to_save.mode != 'L':
            img_to_save = img_to_save.convert('L')
            
        img_to_save.save(jpg_save_path)
        print(f"✅ Saved visualization image to: {jpg_save_path}")


        print("\n--- Output Array Content (First 5 Rows) ---")
        # Print the first few rows of the normalized array
        print(new_image[0, 0:5, :])

# Fashion MNIST labels
K_CATEGORY_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Assuming new_image is a normalized 28x28 numpy array
# 1. Color inversion (black background to match training data)
new_image = 1.0 - new_image
# 2. Horizontal flip (to match training data orientation)
# new_image = np.flip(new_image, axis=2)

# Make prediction
prediction = model.predict(new_image)
predicted_class = np.argmax(prediction[0])
confidence = 100 * np.max(prediction[0])

print(f"Predicted class: {K_CATEGORY_LABELS[predicted_class]} ({confidence:.2f}%)")
