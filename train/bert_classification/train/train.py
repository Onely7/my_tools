# Standard Library
import sys
sys.path.append('..')
# Standard Library
import datetime
import math
import random
# Third Party Library
import hydra
import numpy as np
import torch
import wandb
from configs.config import MyConfig
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)


########################################################################################################################
# setting
########################################################################################################################

# Function to specify seeds for reproducibility all at once
def set_seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# Function to initialize W&B
def wandb_setting(cfg):
    wandb.login()
    config = {
        "model_name": cfg.model.name,
        "data_name": cfg.data.name,
        "epoch": cfg.train.dryrun_epoch if cfg.dry_run else cfg.train.epoch,
        "train_batch_size": cfg.train.train_batch_size,
        "valid_batch_size": cfg.train.valid_batch_size,
        "seed": cfg.train.seed,
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
        "metric": cfg.train.metric,
        "loss": cfg.train.loss,
        "device": cfg.device,
    }
    wandb.init(
        project="nlp-singularity",
        name=cfg.data.name + cfg.model.name + "_" + str(datetime.date.today()),
        config=config
        )

# Function to define optimizer
def make_optimizer(params, name, **kwargs):
    optimizer = torch.optim.__dict__[name](params, **kwargs)
    return optimizer


########################################################################################################################
# train loop
########################################################################################################################

def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

def run(cfg):
    # ensure reproducibility
    set_seed_all(cfg.train.seed)
    # initial W&B settings
    wandb_setting(cfg)

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    # load dataset
    train_dataset = load_from_disk(cfg.data.tokenized.train)
    valid_dataset = load_from_disk(cfg.data.tokenized.valid)
    encoded_train_dataset = load_from_disk(cfg.data.encoded.train)
    encoded_valid_dataset = load_from_disk(cfg.data.encoded.valid)
    # define data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define model
    class_label = train_dataset.features["label"]
    label2id = {label: id for id, label in enumerate(class_label.names)}
    id2label = {id: label for id, label in enumerate(class_label.names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=class_label.num_classes,
        label2id=label2id,  # specify correspondence from label name to ID
        id2label=id2label,  # specify correspondence from ID to label name
    ).to(cfg.device)

    # define optimizer and scheduler
    params = filter(lambda x: x.requires_grad, model.parameters())
    # If you want to rewrite the optimizer, please rewrite it directly
    optimizer = make_optimizer(params, **cfg["train"]["optimizer"])
    num_warmup_steps = math.ceil(train_dataset.num_rows / cfg.train.train_batch_size) * 1
    num_training_steps = math.ceil(valid_dataset.num_rows / cfg.train.train_batch_size) * 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,  # folder to save the results
        per_device_train_batch_size=cfg.train.train_batch_size,  # batch size at training
        per_device_eval_batch_size=cfg.train.valid_batch_size,  # batch size at evaluating
        num_train_epochs=cfg.train.dryrun_epoch if cfg.dry_run else cfg.train.epoch,  # epoch num
        save_strategy="epoch",  # when to save checkpoints
        logging_strategy="epoch",  # when to logging
        evaluation_strategy="epoch",  # timing of evaluation by validation set
        load_best_model_at_end=True,  # load the best model in the valid set after training
        metric_for_best_model=cfg.train.metric,  # metrics to determine the best model
        report_to="wandb",  # report logs to wandb
    )
    # run training
    trainer = Trainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
        optimizers=[optimizer, scheduler],
    )
    trainer.train()




@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: MyConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)

if __name__ == "__main__":
    main()
