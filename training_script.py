from libs.model import TweetMSA, TweetMSAConfig
from libs.dataset_loader import MulTweEmoDataset
from torch import manual_seed
from transformers import Trainer, TrainingArguments

manual_seed(123)

config = TweetMSAConfig(feature_extractor="jinaai/jina-clip-v1", n_layers=2)
model = TweetMSA(config).cuda()

dataset = MulTweEmoDataset.load(seed=123, generate_captions=False)["train"].train_test_split(test_size=0.25)["train"]

print("Processing dataset")
dataset =  model.preprocess_dataset(dataset=dataset)
print("Done!")

args = TrainingArguments(per_device_train_batch_size=16, 
                         resume_from_checkpoint=False, 
                         save_strategy="no", output_dir="./cache/", 
                         num_train_epochs=1,eval_strategy="no", 
                         logging_strategy="no",
                          )

trainer = Trainer(model=model, train_dataset=dataset, args=args, tokenizer=None)

print("Training Model")
trainer.train()
print("Done")
print(trainer.predict(test_dataset=dataset).metrics)