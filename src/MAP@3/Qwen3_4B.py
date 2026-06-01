import os
import joblib
import modal #type: ignore

app = modal.App("math-misconception-qwen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "sentencepiece"
    )
)

volume = modal.Volume.from_name(
    "dataset",
    create_if_missing=False
)

hf_secret = modal.Secret.from_name("hf-secret")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def train():

    import pandas as pd
    import numpy as np
    import torch

    from datasets import load_dataset, Dataset
    from sklearn.preprocessing import LabelEncoder

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments, #type: ignore
        Trainer, #type: ignore
        DataCollatorWithPadding, #type: ignore
    )

    from peft import ( #type: ignore
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    HF_TOKEN = os.environ["HF_TOKEN"]

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    CACHE_DIR = "/data/cache"
    OUTPUT_DIR = "/data/qwen3_4b"

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")

    ds = load_dataset(
        "guldasta/Math_misconception",
        token=HF_TOKEN
    )

    train = ds["train"].to_pandas() #type: ignore
    train = train.head(500) #type: ignore

    train.Misconception = train.Misconception.fillna("NA") #type: ignore

    train["target"] = (
        train.Category
        + ":"
        + train.Misconception
    )

    le = LabelEncoder()

    train["label"] = le.fit_transform( #type: ignore
        train["target"]
    )

    n_classes = len(le.classes_)

    print("Classes:", n_classes)

    idx = (
        train.apply(
            lambda row:
            row.Category.split("_")[0] == "True",
            axis=1,
        )
    )

    correct = train.loc[idx].copy()

    correct["c"] = (
        correct.groupby(
            ["QuestionId", "MC_Answer"]
        )
        .MC_Answer.transform("count")
    )

    correct = (
        correct
        .sort_values("c", ascending=False)
        .drop_duplicates(["QuestionId"])
    )

    correct = correct[
        ["QuestionId", "MC_Answer"]
    ]

    correct["is_correct"] = 1

    train = train.merge(
        correct,
        on=["QuestionId", "MC_Answer"],
        how="left",
    )

    train.is_correct = ( #type: ignore
        train.is_correct.fillna(0)
    )

    def format_input_v2(row):

        correct_text = (
            "Yes"
            if row["is_correct"]
            else "No"
        )

        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"Is Correct Answer: {correct_text}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )

    train["text"] = train.apply(
        format_input_v2,
        axis=1,
    )

    train_df = train[
        ["text", "label"]
    ].copy()

    train_df["label"] = (
        train_df["label"]
        .astype(np.int64)
    )

    train_ds = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        token=HF_TOKEN,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]"}
    )

    model.resize_token_embeddings(
        len(tokenizer)
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.3,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )

    model = prepare_model_for_kbit_training(
        model
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving model...")

    trainer.save_model(
        f"{OUTPUT_DIR}/best"
    )

    tokenizer.save_pretrained(
        f"{OUTPUT_DIR}/best"
    )

    joblib.dump(
        le,
        f"{OUTPUT_DIR}/label_encoder.joblib",
    )

    volume.commit()

    print("Training completed.")
    print("Saved to Modal volume.")


@app.local_entrypoint()
def main():
    train.remote()