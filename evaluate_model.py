import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("✅ Starting model evaluation...")

# === CONFIG ===
model_path = "Blood_Cell.keras"
test_dir = "D:/apsche-project/dataset/images/test_images/"
img_size = (128, 128)
batch_size = 32

# === LOAD MODEL ===
print("📦 Loading model...")
model = tf.keras.models.load_model(model_path)

# === TEST DATA LOADER ===
print("🖼️ Preparing test dataset...")
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === EVALUATE ===
print("🚀 Evaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)

print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print(f"📉 Test Loss: {loss:.4f}")
