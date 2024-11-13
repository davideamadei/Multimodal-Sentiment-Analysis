from libs.dataset_loader import MulTweEmoDataset
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from libs.model import TweetMSAConfig
import torch
import numpy as np
if __name__ == "__main__":
    clip_versions = ["clip_base", "jina", "clip_large"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictions = []

    for clip_version in clip_versions:
        processor = AutoProcessor.from_pretrained(TweetMSAConfig.get_feature_extractor_name(clip_version), trust_remote_code=True)

        model = AutoModel.from_pretrained(TweetMSAConfig.get_feature_extractor_name(clip_version), trust_remote_code=True)
        model.to(device)
        mode="M"
        train, _ = MulTweEmoDataset.load(csv_path="./dataset/train_MulTweEmo.csv", mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=123)
        processed_images= []
        for img in train["img_path"]:
            if isinstance(img, str):
                if img.startswith('http'):
                    response = requests.get(img)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                image = img.convert('RGB')
            else:
                raise ValueError("Unsupported image format")
            processed_images.append(image)

        emotions = MulTweEmoDataset.get_labels()
        emotions.remove("something else")
        candidate_labels = [f"This picture evokes {label}" for label in emotions]

        inputs = processor(images=processed_images, text=candidate_labels, return_tensors="pt", padding=True)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
        predictions.append(predicted_labels)
    
        del model
        del processor

    predictions = np.array(predictions)
    with open(f"zero_shot_predictions", "wb") as f:
        np.save(f, predictions)