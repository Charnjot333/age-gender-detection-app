import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("age_gender_model.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected ❌"

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]

    # Preprocess
    face = cv2.resize(face, (200, 200))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Predict
    preds = model.predict(face)

    age = preds[0][0][0]
    gender = "Male" if preds[1][0][0] > 0.5 else "Female"

    return f"Age: {int(age)} | Gender: {gender}"

# UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),   # ✅ FIXED
    outputs="text",
    title="Age & Gender Detection",
    description="Upload or capture image"
)

app.launch()