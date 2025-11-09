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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    predicted_class: str
    probability: float
    gradcam_png_base64: str | None = None

# Load model once
model = None
class_names = ["NORMAL", "PNEUMONIA"]

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        # Build a tiny untrained fallback to let endpoint run
        base = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(base.input, out)
        print("WARNING: No model.h5 found; using untrained fallback.")

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), with_gradcam: bool = True):
    raw = await file.read()
    arr = preprocess_image(raw)
    prob = float(model.predict(arr, verbose=0).flatten()[0])
    idx = int(prob >= 0.5)
    predicted = class_names[idx]

    resp = PredictResponse(predicted_class=predicted, probability=prob, gradcam_png_base64=None)

    if with_gradcam:
        # Grad-CAM (best effort)
        # find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        if last_conv_layer is None:
            # try nested
            for layer in reversed(model.layers):
                if hasattr(layer, 'layers'):
                    for l2 in reversed(layer.layers):
                        if isinstance(l2, tf.keras.layers.Conv2D):
                            last_conv_layer = l2.name
                            break
                    if last_conv_layer:
                        break

        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)
        heatmap = heatmap.numpy()

        # overlay
        # reconstruct RGB image for overlay
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224,224))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay_bgr = grad_cam_overlay(img_bgr, heatmap)
        b64 = img_to_base64(overlay_bgr)
        resp.gradcam_png_base64 = b64

    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
