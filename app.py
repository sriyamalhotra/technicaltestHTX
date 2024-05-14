from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

import torch
from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
import io

# Defining the FastAPI application
app = FastAPI()

# Load image classification model for gender
gender_processor = AutoProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained(
    "rizvandwiki/gender-classification"
)
gender_model.eval()

# Load sentiment analysis models
star_review_model = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
)
positive_negative_model = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
emotion_detection_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=5,
)

# Load image classification model for general purpose
feature_extractor = AutoProcessor.from_pretrained("microsoft/resnet-101")
resnet_model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-101",
)
resnet_model.eval()


@app.post("/star-review/")
async def star_review(text: str = Form(...)):
    """Endpoint to classify text sentiment into star ratings from 1 to 5."""
    result = star_review_model(text)
    print(result)
    return {"label": result[0]["label"], "score": result[0]["score"]}


@app.post("/positive-negative/")
async def positive_negative(text: str = Form(...)):
    """Endpoint to classify text as positive or negative with percentage scores."""
    result = positive_negative_model(text)
    positive_score = (
        "{:.2f}%".format(result[0]["score"] * 100)
        if result[0]["label"] == "POSITIVE"
        else "{:.2f}%".format((1 - result[0]["score"]) * 100)
    )
    negative_score = (
        "{:.2f}%".format((1 - result[0]["score"]) * 100)
        if result[0]["label"] == "POSITIVE"
        else "{:.2f}%".format(result[0]["score"] * 100)
    )
    return {
        "label": result[0]["label"],
        "positive_score": positive_score,
        "negative_score": negative_score,
    }


@app.post("/emotion-detection/")
async def emotion_detection(text: str = Form(...)):
    """Endpoint to detect multiple emotions from text, returning the top possible emotions with their scores."""
    result = emotion_detection_model(text)
    print(result)
    emotions = [{"label": res["label"], "score": res["score"]} for res in result[0]]
    return {"emotions": emotions}


@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    """Endpoint to classify an uploaded image into one of the 1000 ImageNet classes using ResNet-101."""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = resnet_model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    prediction = logits.softmax(dim=-1).tolist()
    if prediction[0][predicted_label] > 0.7:
        return {
            "predicted_class": resnet_model.config.id2label[predicted_label],
            "Score": prediction[0][predicted_label],
        }
    else:
        return {
            "predicted_class": "Not sure",
            "Closest Class": resnet_model.config.id2label[predicted_label],
            "Score": prediction[0][predicted_label],
        }


@app.post("/classify-gender/")
async def classify_gender(file: UploadFile = File(...)):
    """Endpoint to classify gender from an uploaded image using a pretrained model."""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    try:
        inputs = gender_processor(images=image, return_tensors="pt")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "message": "Invalid image format. Kindly input jpg format image files. Please try again with a different image."
            },
        )
    with torch.no_grad():
        logits = gender_model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    prediction = logits.softmax(dim=-1).tolist()
    print(prediction)
    if prediction[0][0] > 0.7:
        return {"Predicted_gender": "Female", "Score": prediction[0][0]}
    elif prediction[0][1] > 0.7:
        return {"Predicted_gender": "Male", "Score": prediction[0][1]}
    else:
        return {"Predicted_gender": "Not sure"}


# Running the fastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        port=8000,
        reload=True,
    )
