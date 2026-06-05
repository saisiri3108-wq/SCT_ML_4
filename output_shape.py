from tensorflow.keras.models import load_model

model = load_model("asl_hand_gesture_model.keras")

print(model.output_shape)
model.summary()