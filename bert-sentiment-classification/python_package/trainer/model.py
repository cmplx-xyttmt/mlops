from transformers import AutoModelForSequenceClassification
from trainer import metadata


def create(num_labels):
    """create the model by loading a pretrained model or define your own
    Args:
        num_labels: number of target labels
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        metadata.PRETRAINED_MODEL_NAME,
        num_labels=num_labels
    )

    return model
