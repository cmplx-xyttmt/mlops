import os

from google.cloud import storage
from transformers import AutoTokenizer
from datasets import load_dataset
from trainer import metadata


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.PRETRAINED_MODEL_NAME,
        use_fast=True
    )

    # Tokenize the texts
    tokenizer_args = (
        (examples['text'],)
    )

    result = tokenizer(*tokenizer_args,
                       padding='max_length',
                       max_length=metadata.MAX_SEQ_LENGTH,
                       truncation=True)

    label_to_id = metadata.TARGET_LABELS

    if label_to_id is not None and "label" in examples:
        result["label"] = [label_to_id[l] for l in examples["label"]]

    return result


def load_data(args):
    """Loads the data into two different data loaders. (Train, Test)

        Args:
            args: arguments passed to the python script
    """
    dataset = load_dataset(metadata.DATASET_NAME)

    dataset = dataset.map(preprocess_function,
                          batched=True,
                          load_from_cache_file=True)

    train_dataset, test_dataset = dataset["train"], dataset["test"]

    return train_dataset, test_dataset


def save_model(args):
    """Saves the model to Google Cloud Storage or local file system
    Args:
        args: contains name for saved model
    """
    scheme = 'gs://'
    if args.job_dir.startswith(scheme):
        job_dir = args.job_dir.split("/")
        bucket_name = job_dir[2]
        object_prefix = "/".join(job_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = f'{object_prefix}/{args.model_name}'
        else:
            model_path = f'{args.model_name}'

        bucket = storage.Client().bucket(bucket_name)
        local_path = os.path.join("/tmp", args.model_name)
        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(f"To save model files in GCS bucket, please specify job_dir starting with gs://")
