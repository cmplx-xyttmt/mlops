# Task type can be either 'classification', 'regression' or 'custom'
# based on the target feature in the dataset
TASK_TYPE = 'classification'

DATASET_NAME = 'imdb'

PRETRAINED_MODEL_NAME = 'bert-base-cased'

TARGET_LABELS = {1:1, 0:0, -1:0}

MAX_SEQ_LENGTH = 128
