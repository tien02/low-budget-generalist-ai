import torch
import config
import uvicorn
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


model, preprocess = create_model_from_pretrained(config.CLIP_CKPT)
tokenizer = get_tokenizer(config.CLIP_CKPT)
model.eval()

template = 'this is a photo of '

diseaseList = [
    'brain MRI',
    'covid line chart',
    'pie chart',
    'broken bone X-ray'
    'bone X-ray',
    'chest X-ray',
    'benign histopathology',
    'adenocarcinoma histopathology',
    'immunohistochemistry histopathology',
    'hematoxylin and eosin histopathology',
    'squamous cell carcinoma histopathology',
    'normal lymph node',
    'lymph node metastasis',
    'lung adenocarcinoma',
    'normal lung tissue',
    'lung squamous cell carcinomas',
    'colon adenocarcinomas',
    'normal colonic tissue',
    'tumor infiltrating lymphocytes',
    'pneumonia lung',
    ]

context_length = 256
textsTensor = tokenizer([template + l for l in diseaseList], context_length=context_length)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def readImageAsTensor(image_file)->torch.tensor:
    image = Image.open(image_file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image_tensor = torch.stack([preprocess(image)])

    return image_tensor


def retrieval(image_tensor):
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, textsTensor)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        logits = logits.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

        return logits, sorted_indices
    

def find_symptom(image_file):
    image_tensor = readImageAsTensor(image_file)

    logits, sorted_indices = retrieval(image_tensor)

    top_1_indices = sorted_indices[0, 0]

    return {
        "label": diseaseList[top_1_indices]
    }


@app.get('/')
def home():
    return {
        "message": "This is BioMedCLIP for zero-shot medical-image-classfication"
    }


@app.post('/disease_information')
async def give_symptoms_information(
    img_files: UploadFile = File(..., description="Upload Image")
):
    predicted_symtom = find_symptom(img_files.file)
    
    return {
        'result': predicted_symtom
    }


if __name__ == '__main__':
    uvicorn.run("clip:app", host="0.0.0.0", port=config.CLIP_PORT)