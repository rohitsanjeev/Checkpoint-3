from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io

# Define Pydantic model for prediction output
class PredictionItem(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]

# Load YOLO model
model = YOLO("detect/device_detection2/weights/best.pt")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict/", response_model=list[PredictionItem])
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Run YOLO prediction
    results = model.predict(img)

    predictions = []
    for r in results:
        for box in r.boxes:
            item = PredictionItem(
                class_name=model.names[int(box.cls)],
                confidence=float(box.conf) * 100,  # convert to percentage
                bbox=box.xyxy[0].tolist()          # convert tensor to list
            )
            predictions.append(item)

    return predictions
