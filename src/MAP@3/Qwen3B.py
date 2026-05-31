import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal #type: ignore

app = modal.App("example-unsloth-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_READ": "/model_cache"})
)

with train_image.imports():
    # unsloth must be first!
    import unsloth #type: ignore  # noqa: F401,I001
    import datasets
    import torch
    import wandb
    from transformers import TrainingArguments #type: ignore
    from trl import SFTTrainer #type: ignore
    from unsloth import FastLanguageModel #type: ignore
    from unsloth.chat_templates import standardize_sharegpt #type: ignore


model_cache_volume = modal.Volume.from_name(
    "unsloth-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "unsloth-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "unsloth-checkpoints", create_if_missing=True
)

GPU_TYPE = "T4"
TIMEOUT_HOURS = 6
MAX_RETRIES = 3

CONVERSATION_COLUMN = "conversations"  # ShareGPT format column name
TEXT_COLUMN = "text"  # Output column for formatted text
TRAIN_SPLIT_RATIO = 0.9  # 90% train, 10% eval split
PREPROCESSING_WORKERS = 2  # Number of workers for dataset processing


def format_chat_template(examples, tokenizer):
    texts = []
    for conversation in examples[CONVERSATION_COLUMN]:
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {TEXT_COLUMN: texts}


def load_or_cache_dataset(config: "TrainingConfig", paths: dict, tokenizer): #type: ignore
    dataset_cache_path = paths["dataset_cache"]

    if dataset_cache_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        train_dataset = datasets.load_from_disk(dataset_cache_path / "train")
        eval_dataset = datasets.load_from_disk(dataset_cache_path / "eval")
    else:
        print(f"Downloading and processing dataset: {config.dataset_name}")

        # Load and standardize the dataset format
        dataset = datasets.load_dataset(config.dataset_name, split="train")
        dataset = standardize_sharegpt(dataset)

        # Split into training and evaluation sets with fixed seed for reproducibility
        dataset = dataset.train_test_split(
            test_size=1.0 - TRAIN_SPLIT_RATIO, seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Apply chat template formatting to convert conversations to text
        print("Formatting datasets with chat template...")
        train_dataset = train_dataset.map(
            lambda examples: format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )

        eval_dataset = eval_dataset.map(
            lambda examples: format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

        # Cache the processed datasets for future runs
        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")

        # Commit the dataset cache to the volume
        dataset_cache_volume.commit()

    return train_dataset, eval_dataset



