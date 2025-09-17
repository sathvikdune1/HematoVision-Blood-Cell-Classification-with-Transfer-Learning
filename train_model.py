import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIGURATION ===
img_width, img_height = 128, 128
batch_size = 32
epochs = 10
num_classes = 4  # 4 types of blood cells
dataset_path = "D:/apsche-project/dataset/train images/"

# === DATA GENERATORS ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# === MODEL ARCHITECTURE ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Must match number of folders
])

# === COMPILE ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAINING ===
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# === SAVE MODEL ===
model.save("Blood_Cell.keras")
print("Model saved as Blood_Cell.keras")
