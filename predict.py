from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = load_model("asl_hand_gesture_model.keras")

# Class names for ASL dataset
class_names = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]

# Load test image
img_path = r"test.jpg\Screenshot 2026-06-05 165335.png"

img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)

# Normalize image
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get prediction
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = np.max(prediction) * 100

# Print result
print("Predicted Gesture:", predicted_class)
print("Confidence:", round(confidence, 2), "%")

# Display image
display_img = image.load_img(img_path)

plt.imshow(display_img)
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()