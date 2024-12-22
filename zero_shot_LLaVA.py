from libs.dataset_loader import MulTweEmoDataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from libs.model import TweetMSAConfig
import torch
import numpy as np
import argparse
import regex as re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='zero shot image classification',
        description='Zero shot classification on the images only. Predictions are done with clip base, jina clip and clip large',
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-o", "--output", type=str, default="zero-shot-predictions.np")
    parser.add_argument("--binary_prediction", action="store_true", help="predict each label on its own")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = []
    labels = MulTweEmoDataset.get_labels()
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=False
    )
    model.to(device)
    
    mode="M"
    test_data, _ = MulTweEmoDataset.load(csv_path="./dataset/test_MulTweEmo.csv", mode=mode, drop_something_else=True, force_override=True, test_split=None, seed=123)

    predictions = np.zeros(len(test_data), len(labels))

    if not args.binary_prediction:
        prompt_format = lambda text: f"The image is paired with this text: \"{text}\". Considering both image and text, choose which emotions are most elicited among this list: {labels}. Answer with only the list of chosen emotions."

        outputs = []
        with torch.no_grad():
            for i, row in test_data.iterrows():

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_format(row["tweet"])},
                            ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                img = row["img_path"]
                if isinstance(img, str):
                    if img.startswith('http'):
                        response = requests.get(img)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        img = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    img = img.convert('RGB')
                else:
                    raise ValueError("Unsupported image format")

                inputs = processor(images=img, text=prompt, return_tensors="pt", padding=True).to(device)

                generate_ids = model.generate(**(inputs))
                outputs.append(processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

        model_predictions = []
        translate_table = dict.fromkeys(map(ord, '\n[]\' '), None)
        for item in outputs:
            item = item.translate(translate_table)
            predictions.append(item.split(","))        

        n_labels = len(labels)
        label2id = MulTweEmoDataset.get_label2id()
        for i in range(len(model_predictions)):
            for label in item:
                if label in labels:
                    predictions[i][label2id[label]] = 1

    else:
        prompt_format = lambda text, emotion: f"The image is paired with this text: \"{text}\". When looking at both image and text, is the emotion evoked \"{emotion}\"? Answer with Yes or no."

        predictions = np.zeros((len(test_data), len(labels)))
        pos_pattern = re.compile("yes", re.I)
        neg_pattern = re.compile("no", re.I)
        error_counter = 0
        with torch.no_grad():    
            for i, row in test_data.iterrows():
                img = row["img_path"]
                if isinstance(img, str):
                    if img.startswith('http'):
                        response = requests.get(img)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        img = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    img = img.convert('RGB')
                else:
                    raise ValueError("Unsupported image format")
                
                for j, emotion in enumerate(labels):
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_format(row["tweet"], emotion)},
                                ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=img, text=prompt, return_tensors="pt", padding=True).to(device)
                    generate_ids = model.generate(**(inputs))
                    output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    if pos_pattern.match(output):
                        predictions[i][j] = 1
                    elif not neg_pattern.match(output):
                        error_counter += 1
        print(f"The model gave non valid outputs {error_counter} times")
                
    with open(args.output, "wb") as f:
        np.save(f, predictions)