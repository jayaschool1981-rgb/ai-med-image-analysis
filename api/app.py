import os, io, uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from api.utils.inference import preprocess_image, grad_cam_overlay, img_to_base64

APP_TITLE = "AI-Powered Medical Image Analysis API"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/v1/model.h5")

app = FastAPI(title=APP_TITLE)

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    predicted_class: str
    probability: float
    gradcam_png_base64: str | None = None

model = None
class_names = ["NORMAL", "PNEUMONIA"]

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        base = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(base.input, out)
        print("⚠️ Using fallback model (no trained model found)")


def safe_make_gradcam(rawbytes: bytes) -> str | None:
    """
    Grad-CAM but isolated from the main model graph — no graph-disconnected error.
    """
    try:
        if model is None:
            return None

        # Read & preprocess
        img = Image.open(io.BytesIO(rawbytes)).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)

        # Extract the EfficientNet submodel safely
        try:
            base_model = model.get_layer("efficientnetb0")
        except:
            print("⚠️ efficientnetb0 not found in model.")
            return None

        # Build a grad-capable model only for the EfficientNet part
        last_conv_layer = base_model.get_layer("top_conv")
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(conv_outputs[0], weights.numpy())
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        img_bgr = cv2.cvtColor((arr[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = grad_cam_overlay(img_bgr, cam)
        return img_to_base64(overlay)
    except Exception as e:
        print("❌ Grad-CAM failed:", e)
        return None


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), with_gradcam: bool = True):
    raw = await file.read()
    arr = preprocess_image(raw)

    prob = float(model.predict(arr, verbose=0).flatten()[0])
    idx = int(prob >= 0.5)
    predicted = class_names[idx]

    gradcam_b64 = None
    if with_gradcam:
        gradcam_b64 = safe_make_gradcam(raw)

    return PredictResponse(
        predicted_class=predicted,
        probability=prob,
        gradcam_png_base64=gradcam_b64
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
