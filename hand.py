# ==========================================
# Hand Gesture Recognition using ASL Dataset
# ==========================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# ==========================================
# DATASET PATH
# ==========================================

dataset_path = r"asl_alphabet_train\asl_alphabet_train"

# ==========================================
# DATA PREPROCESSING
# ==========================================

data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
 
train_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ==========================================
# CHECK CLASSES
# ==========================================

print("\nNumber of Classes:", train_data.num_classes)
print("\nClass Labels:")
print(train_data.class_indices)

# ==========================================
# BUILD CNN MODEL
# ==========================================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,3)
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(
    Dense(
        train_data.num_classes,
        activation='softmax'
    )
)

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# ==========================================
# EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(validation_data)

print("\nValidation Accuracy:",
      round(accuracy * 100, 2), "%")

# ==========================================
# SAVE MODEL
# ==========================================

model.save("asl_hand_gesture_model.keras")

print("\nModel saved successfully!")

# ==========================================
# CHECK OUTPUT SHAPE
# ==========================================

print("\nModel Output Shape:")
print(model.output_shape)

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.show()

# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Training Loss",
    "Validation Loss"
])

plt.show()