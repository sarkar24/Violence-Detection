from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Initialize Flask app
app = Flask(__name__)

# Load the trained violence detection model
model = tf.keras.models.load_model("model.h5")  # Ensure model.h5 is in the same directory

# Open the camera
camera = cv2.VideoCapture(0)  # 0 for default webcam

# Frame buffer to store 16 consecutive frames (rolling window)
frame_buffer = deque(maxlen=16)

# Stability settings
last_predictions = deque(maxlen=10)  # Store last 10 predictions
violence_threshold = 0.8  # Only detect violence if probability > 80%
min_violence_frames = 5  # Violence must be detected in at least 5 of the last 10 frames

# Function to preprocess the frame before feeding it to the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to match model input
    frame = frame / 255.0  # Normalize
    return frame

# Function to generate frames for the video feed
def generate_frames():
    global frame_buffer, last_predictions

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame
            processed_frame = preprocess_frame(frame)
            frame_buffer.append(processed_frame)  # Add frame to buffer

            # Ensure we have 16 frames to make a valid prediction
            if len(frame_buffer) == 16:
                input_sequence = np.array(frame_buffer).reshape(1, 16, 64, 64, 3)

                # Model prediction
                prediction = model.predict(input_sequence)[0][0]
                last_predictions.append(prediction)

                # Decision: Only display "Violence Detected" if at least `min_violence_frames` out of the last 10 predictions are above the threshold
                violence_count = sum(1 for p in last_predictions if p > violence_threshold)
                if violence_count >= min_violence_frames:
                    label = "Violence Detected"
                    color = (0, 0, 255)  # Red for violence
                else:
                    label = "No Violence"
                    color = (0, 255, 0)  # Green for no violence

            else:
                label = "Analyzing..."
                color = (255, 255, 0)  # Yellow while collecting frames

            # Display prediction on the frame
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
