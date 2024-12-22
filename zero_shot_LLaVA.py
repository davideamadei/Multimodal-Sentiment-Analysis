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

#    index = 0
#    train = train.iloc[index:index+1]
    prompt_format = lambda text: f"The image is paired with this text: \"{text}\". Considering both image and text, choose which emotions are most elicited among this list: \
{labels}. Answer with only the list of chosen emotions."
    prompts = []

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
        prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))
    processed_images= []
    for img in test_data["img_path"]:
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
    inputs = []
    for i in range(len(prompts)):
        inputs.append(processor(images=processed_images[i], text=prompts[i], return_tensors="pt", padding=True).to(device))
    outputs = []
    with torch.no_grad():
        for i in range(len(inputs)):
    #       for i in range(1):
                generate_ids = model.generate(**(inputs[i]))
    #           generate_ids = model.generate(input_ids=inputs["input_ids"][i:i+1], attention_mask=inputs["attention_mask"][i:i+1], pixel_values=inputs["pixel_values"][i:i+1], image_sizes=inputs["image_sizes"][i:i+1])
                outputs.append(processor.decode(generate_ids[0, inputs[i]["input_ids"].shape[1]:], skip_special_tokens=True))
    # Generate
#       generate_ids = model.generate(**inputs, max_new_tokens=100)
#       outputs =  processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    predictions = []
    translate_table = dict.fromkeys(map(ord, '\n[]\' '), None)
    for item in outputs:
        item = item.translate(translate_table)
        predictions.append(item.split(","))
    print(predictions)
    id_predictions = []
    n_labels = len(labels)
    label2id = MulTweEmoDataset.get_label2id()
    for item in predictions:
        pred = [0] * n_labels
        for label in item:
            pred[label2id[label]] = 1
        id_predictions.append(pred)
    print(id_predictions)

    id_predictions = np.array(id_predictions)
    with open(args.output, "wb") as f:
        np.save(f, id_predictions)
