from libs.dataset_loader import MulTweEmoDataset
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from libs.model import TweetMERConfig
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='zero shot image classification',
        description='Zero shot classification on the images only. Predictions are done with clip base, jina clip and clip large',
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-o", "--output", type=str, default="zero-shot-predictions.np")
    parser.add_argument("-p", "--prompt", type=str, default="This picture evokes")
    args = parser.parse_args()

    feature_extractors = ["base", "jina", "large", "siglip"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictions = []

    for feature_extractor in feature_extractors:
        processor = AutoProcessor.from_pretrained(TweetMERConfig.get_feature_extractor_name(feature_extractor), trust_remote_code=True)

        model = AutoModel.from_pretrained(TweetMERConfig.get_feature_extractor_name(feature_extractor), trust_remote_code=True)
        model.to(device)
        mode="M"
        train, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=123)
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
        candidate_labels = [f"{args.prompt} {label}" for label in emotions]

        inputs = processor(images=processed_images, text=candidate_labels, return_tensors="pt", padding=True)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
        predictions.append(predicted_labels)
    
        del model
        del processor

    predictions = np.array(predictions)
    with open(args.output, "wb") as f:
        np.save(f, predictions)