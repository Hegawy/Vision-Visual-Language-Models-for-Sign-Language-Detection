from flask import Flask, render_template, Response
import cv2
import torch
import open_clip
import os
from collections import deque
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from io import BytesIO

app = Flask(__name__)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔌 Using device: {device}")

# --- Load CLIP ---
clip_model, _, preprocess_clip = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device).eval()

# --- Define classifier ---
class SignClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        logits = self.fc(x)
        return logits

# --- Load checkpoint ---
checkpoint = torch.load('asl_clip_finetuned.pth', map_location=device)
class_names = checkpoint['class_names']
model = SignClassifier(clip_model, len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])

# --- Prediction Buffer ---
prediction_buffer = deque(maxlen=10)
confidence_threshold = 0.0

# --- MediaPipe Hands Detector ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=0.5)

# --- Webcam Capture for Flask ---
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror horizontally for natural interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        # --- Skip if no hand is detected ---
        if not results.multi_hand_landmarks:
            continue

        # --- Preprocess and predict ---
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top1_prob, top1_index = torch.max(probs, dim=1)
            top1_label = class_names[top1_index.item()]
            top1_conf = top1_prob.item()

        prediction_buffer.append((top1_label, top1_conf))

        # --- Smoothing ---
        votes = [p[0] for p in prediction_buffer]
        avg_prob = sum([p[1] for p in prediction_buffer]) / len(prediction_buffer)
        smoothed_class = max(set(votes), key=votes.count)

        # --- Show prediction if confident ---
        if avg_prob > confidence_threshold:
            text = smoothed_class
            font_scale = 3
            font_thickness = 6
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            x = int((frame.shape[1] - text_width) / 2)
            y = 100

            # Background box
            cv2.rectangle(frame, (x - 30, y - text_height - 30), (x + text_width + 30, y + 30), (255, 255, 255), -1)
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 128, 255), font_thickness, cv2.LINE_AA)

            # Confidence score
            cv2.putText(frame, f"Confidence: {avg_prob*100:.1f}%", (x, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2, cv2.LINE_AA)
        
        # Convert frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)  # Use port 5001 or another unused port

