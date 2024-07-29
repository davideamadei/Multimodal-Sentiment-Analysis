from libs.model import TweetMSA, TweetMSAConfig
from libs.dataset_loader import MulTweEmoDataset
from torch import manual_seed
from transformers import Trainer, TrainingArguments
import os
import sklearn.metrics as skm
import argparse

def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = y_pred > 0.5
    measure_dict = {}

    measure_dict["exact_match"] = skm.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    measure_dict["recall"] = skm.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    measure_dict["precision"] = skm.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    measure_dict["f1_score"] = skm.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    return measure_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
	parser.add_argument('-c', '--clip-version', choices=["base", "jina"], type=str, help='clip version for feature extraction')
	args = parser.parse_args()

	clip = args.clip_version
      
	if clip == "base":
		clip_model = "openai/clip-vit-base-patch32"
	elif clip == "jina":	
		clip_model = "jinaai/jina-clip-v1"

	manual_seed(123)

	os.environ["WANDB_PROJECT"] = "msa_thesis"

	config = TweetMSAConfig(feature_extractor=clip_model, n_layers=2)
	model = TweetMSA(config).cuda()
	dataset = MulTweEmoDataset.load(seed=123, generate_captions=False)["train"].train_test_split(test_size=0.25, seed=123)
	val_dataset = dataset["test"]
	dataset = dataset["train"]
	print("Processing dataset")
	dataset =  model.preprocess_dataset(dataset=dataset)
	val_dataset = model.preprocess_dataset(dataset=val_dataset)
	print("Done!")


	args = TrainingArguments(
							report_to="wandb",
							run_name=f"{clip}_clip",
							per_device_train_batch_size=16, 
							resume_from_checkpoint=True,
							save_total_limit=5,

							save_strategy="epoch", 
							output_dir=f"./.ckp/{clip}", 
							num_train_epochs=10,
							eval_strategy="epoch", 
							logging_strategy="no",
							logging_steps=1,
							save_safetensors=False if clip=="jina" else True
							)

	trainer = Trainer(
					model=model,
					train_dataset=dataset,
					eval_dataset=val_dataset,
					args=args,
					tokenizer=None,
					compute_metrics=compute_metrics,
				
					)

	print("Training Model")
	trainer.train()
	print("Done")

	results = trainer.predict(test_dataset=dataset)
	val_results = trainer.predict(test_dataset=val_dataset)

	print(compute_metrics((results.predictions, dataset["labels"])))
	print(compute_metrics((val_results.predictions, val_dataset["labels"])))
      