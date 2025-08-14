import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from datetime import datetime

st.set_page_config(
    page_title="Violence Detection System",
    page_icon="🚨",
    layout="wide"
)

# --- Model Loading ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Loads the pre-trained CNN model and its weights."""
    try:
        with open("cnn_model_224.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("cnn_model_224.h5")
        print("✅ Model loaded successfully")
        return model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'cnn_model_224.json' and 'cnn_model_224.h5' are present.")
        return None

model = load_model()
labels = {0: 'Accident', 1: 'Fight', 2: 'Fire', 3: 'Snatch', 4: 'Normal'}

# --- Session State Initialization ---
# This is to store alerts across reruns.
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# --- WebRTC Video Processing ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.predicted_label = "Normal"
        self.prediction_interval = 5  # Predict every 5 frames

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Receives a video frame from the browser, processes it, and returns the modified frame.
        """
        # Convert the frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1

        # Perform prediction only at the specified interval to save resources
        if self.frame_count % self.prediction_interval == 0 and model is not None:
            # Preprocess the frame for the model
            processed_img = self._preprocess_image(img)
            
            # Make a prediction
            prediction = model.predict(processed_img, verbose=0)
            label_index = np.argmax(prediction)
            self.predicted_label = labels[label_index]

            # If violence is detected, add an alert
            if self.predicted_label != 'Normal':
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert_message = f"🚨 **{self.predicted_label.upper()}** detected at {timestamp}"
                st.session_state.alerts.insert(0, alert_message) # Insert at the beginning for newest first

        # Draw the predicted label on the frame
        cv2.putText(
            img,
            f"Status: {self.predicted_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 0, 255) if self.predicted_label != 'Normal' else (0, 255, 0), # Red for alert, Green for normal
            2,  # Thickness
        )

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Resizes and normalizes the image for the model."""
        image = cv2.resize(image, (224, 224))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0) # Add batch dimension
        return image

# --- Streamlit UI ---
st.title("Live Violence Detection System")
st.markdown("This application uses a pre-trained CNN to detect violent activities from a live webcam feed.")

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Webcam Feed")
    # The main component that handles the webcam stream
    webrtc_ctx = webrtc_streamer(
        key="violence-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    if not webrtc_ctx.state.playing:
        st.info("Click the 'START' button to begin video streaming.")

with col2:
    st.header("Admin Alert Panel")
    
    if st.button("Clear All Alerts"):
        st.session_state.alerts = []

    if not st.session_state.alerts:
        st.success("System is running. No alerts at the moment. ✅")
    else:
        for alert in st.session_state.alerts:
            st.warning(alert)
