import modal #type: ignore

app = modal.App("amc24-qwen25vl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "qwen-vl-utils",
        "pillow"
    )
)

hf_secret = modal.Secret.from_name("hf-secret")

volume = modal.Volume.from_name(
    "dataset",
    create_if_missing=False,
)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 24,
    image=image,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def train():

    import os
    import torch

    from datasets import load_dataset

    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        TrainingArguments, #type: ignore
        BitsAndBytesConfig, #type: ignore
    )

    from peft import ( #type: ignore
        LoraConfig,
        prepare_model_for_kbit_training,
    )

    from trl import SFTTrainer #type: ignore

    CACHE_DIR = "/data/cache"
    OUTPUT_DIR = "/data/qwenvl"

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    HF_TOKEN = os.environ["HF_TOKEN"]

    print("Loading dataset...")

    ds = load_dataset( #type: ignore
        "guldasta/amc24", #type: ignore
        token=HF_TOKEN,
    )["train"]

    print(ds)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    from PIL import Image
    import io

    def load_hf_image(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, dict):
            return Image.open(
                io.BytesIO(img["bytes"])
                ).convert("RGB")
        raise ValueError(
        f"Unsupported image type: {type(img)}"
        )

    def format_example(example):
        image = load_hf_image(example["img_path"]) # actually PIL image
        prompt = (
        f"What is the {example['entity_name']} shown in this product image?\n\n"
        f"Return only the value and unit exactly as found or inferred from the image.\n"
        f"Output format: <number> <unit>"
        )
        return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(example["entity_value"]),
                    }
                ],
            },
        ]
    }

    ds = ds.map(format_example) #type: ignore

    print("Dataset formatted.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    def collate_fn(examples):

        texts = [
            processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in examples
        ]

        images = [
            ex["messages"][0]["content"][0]["image"]
            for ex in examples
        ]

        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=15,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
        data_collator=collate_fn,
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(
        f"{OUTPUT_DIR}/best"
    )

    processor.save_pretrained(
        f"{OUTPUT_DIR}/best"
    )

    volume.commit()

    print("Done.")
    print(f"Saved to {OUTPUT_DIR}/best")


@app.local_entrypoint()
def main():
    train.remote()