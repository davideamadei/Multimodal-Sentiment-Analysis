from libs.dataset_loader import MulTweEmoDataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from libs.model import TweetMSAConfig
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
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = []

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )
    model.to(device)
    
    mode="M"
    train, _ = MulTweEmoDataset.load(csv_path="./dataset/MulTweEmo.csv", mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=123)

    train = train.head(5)

    prompt_format = lambda text: f"The image is paired with this text: {text}. Considering both image and text, choose which emotions are most elicited among this list: \
        [anger, anticipation, disgust, fear, joy, neutral, sadness, surprise, trust]. Answer with only the list of chosen emotions."
    prompts = []

    for i, row in train.iterrows():
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_format(row["tweet"])},
                    ],
            },
        ]
        prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))

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

    inputs = processor(images=processed_images, text=prompts, return_tensors="pt", padding=True)
    inputs.to(device)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=30)
    processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)