from libs.model import TweetMSA, TweetMSAConfig
from libs.dataset_loader import MulTweEmoDataset
from torch import manual_seed
from transformers import Trainer, TrainingArguments
import os
import sklearn.metrics as skm
import argparse
from datasets import Dataset

def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = y_pred > 0.5
    metrics_dict = {}

    metrics_dict["exact_match"] = skm.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    metrics_dict["recall"] = skm.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["precision"] = skm.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    metrics_dict["f1_score"] = skm.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0)
    return metrics_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        prog='train_msa',
        description='Train MSA model',
    )
	parser.add_argument('-c', '--clip-version', choices=["base", "jina"], type=str, help='clip version for feature extraction')
	parser.add_argument("--no-report", action="store_true", help="disables reporting results to wandb")
	# parser.add_argument("--freeze_weights", action="store_false", help="freezes weights of feature extractor")
	# parser.add_argument("--append_captions", action="store_false", help="append auto-generated captions to tweet")
	args = parser.parse_args()

	clip = args.clip_version
      
	if clip == "base":
		clip_model = "openai/clip-vit-base-patch32"
	elif clip == "jina":	
		clip_model = "jinaai/jina-clip-v1"

	report = "wandb"
	if args.no_report:
		report="none"
	manual_seed(123)

	os.environ["WANDB_PROJECT"] = "msa_thesis"

	config = TweetMSAConfig(feature_extractor=clip_model, n_layers=2)
	model = TweetMSA(config).cuda()
	dataset = MulTweEmoDataset.load(seed=123, generate_captions=False)["train"].train_test_split(test_size=0.25, seed=123)
	val_dataset = dataset["test"].select(range(10))
	dataset = dataset["train"].select(range(10))
	print("Processing dataset")
	dataset =  model.preprocess_dataset(dataset=dataset)
	val_dataset = model.preprocess_dataset(dataset=val_dataset)
	print("Done!")
	
	args = TrainingArguments(
							report_to="none",
							run_name=f"{clip}_clip",

							per_device_train_batch_size=16, 
							resume_from_checkpoint=True,
							save_total_limit=5,
							save_strategy="epoch", 
							output_dir=f"./.ckp/{clip}", 
							
							eval_strategy="epoch", 
							
							logging_steps=2,
							
							# necessary to save models
							save_safetensors=False,
							bf16=True,

							num_train_epochs=10,
							warmup_steps=50,

							# disable_tqdm=True,
							# torch_compile=True,
							push_to_hub=False,
							load_best_model_at_end=True,
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

	print(skm.classification_report(val_dataset["labels"], val_results.predictions > 0.5))
      