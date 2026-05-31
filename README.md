# Modal
Scripts for finetuning on modal server

# SetUp

```
modal setup
modal profile current
modal profile list

```

# Creating Volumes

```
modal volume create datasets
modal volume list

```

# Push Local Files to Volumes

## Put Entire Directory (/data)
```
modal volume put datasets data/ /data
```
## Put Specific files
```
modal volume put datasets train.csv /train.csv
modal volume put datasets test.csv /test.csv
```

# Create Secrets

```
modal secret create hf-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
modal secret list

```

# Executing Modal Scripts

```
modal run train_modal.py
modal run --detach train_modal.py
```

# Check Saved Artifacts

```
modal volume ls dataset /qwen3_4b
```

# Download Artifacts

```
modal volume get dataset \
/qwen3_4b/best \
./best_model
```

