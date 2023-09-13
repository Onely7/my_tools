# Standard Library
import sys
sys.path.append('..')
# Standard Library
import random
# Third Party Library
import hydra
import numpy as np
import torch
from configs.config import MyConfig
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
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


########################################################################################################################
# create Dataset
########################################################################################################################

class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        # load datasets
        self.dataset = load_dataset(self.cfg.data.name)


    # function to tokenize texts
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], max_length=512)


    # create train and valid dataset
    def setup_train_val_datasets(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function)
        if self.cfg.dry_run:
            train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
            valid_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(20))
        else:
            train_dataset = tokenized_datasets["train"].shuffle(seed=42)
            valid_dataset = tokenized_datasets["test"].shuffle(seed=42)
        return train_dataset, valid_dataset


    # assign answer "label" to the tokenized data
    def preprocess_text_classification(self, example):
        encoded_example = self.tokenizer(example["text"], max_length=512)
        encoded_example["labels"] = example["label"]
        return encoded_example


def create_dataset(cfg):
    # ensure reproducibility
    set_seed_all(cfg.train.seed)
    # create Dataset instance
    dataset = Dataset(cfg)
    # create Dataset
    train_dataset, valid_dataset = dataset.setup_train_val_datasets(cfg.dry_run)
    # save datasets
    train_dataset.save_to_disk(f"../data/{cfg.data.name}_train_tokenized_dryrun" if cfg.dry_run else f"../data/{cfg.data.name}_train_tokenized")
    valid_dataset.save_to_disk(f"../data/{cfg.data.name}_valid_tokenized_dryrun" if cfg.dry_run else f"../data/{cfg.data.name}_valid_tokenized")
    # encode datasets
    encoded_train_dataset = train_dataset.map(
        dataset.preprocess_text_classification,
        remove_columns=train_dataset.column_names,
    )
    encoded_valid_dataset = valid_dataset.map(
        dataset.preprocess_text_classification,
        remove_columns=valid_dataset.column_names,
    )
    # save datasets
    encoded_train_dataset.save_to_disk(f"../data/{cfg.data.name}_train_encoded_dryrun" if cfg.dry_run else f"../data/{cfg.data.name}_train_encoded")
    encoded_valid_dataset.save_to_disk(f"../data/{cfg.data.name}_valid_encoded_dryrun" if cfg.dry_run else f"../data/{cfg.data.name}_valid_encoded")




@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: MyConfig):
    print(OmegaConf.to_yaml(cfg))
    create_dataset(cfg)

if __name__ == "__main__":
    main()
