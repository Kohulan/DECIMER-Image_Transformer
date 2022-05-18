import os
import sys
import tensorflow as tf

import pickle
from selfies import decoder
import Transformer_decoder
import Efficient_Net_encoder
import config

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load assets
HERE = os.path.dirname(os.path.abspath(__file__))
tokenizer = pickle.load(
    open(os.path.join(HERE, "tokenizer_Isomeric_SELFIES.pkl"), "rb")
)
max_length = pickle.load(
    open(os.path.join(HERE, "max_length_Isomeric_SELFIES.pkl"), "rb")
)

# Image partameters
IMG_EMB_DIM = (10, 10, 232)
IMG_EMB_DIM = (IMG_EMB_DIM[0] * IMG_EMB_DIM[1], IMG_EMB_DIM[2])
IMG_SHAPE = (299, 299, 3)
PE_INPUT = IMG_EMB_DIM[0]
IMG_SEQ_LEN, IMG_EMB_DEPTH = IMG_EMB_DIM
D_MODEL = IMG_EMB_DEPTH

# Network parameters
N_LAYERS = 4
D_MODEL = 512
D_FF = 2048
N_HEADS = 8
DROPOUT_RATE = 0.1

# Misc
MAX_LEN = max_length
VOCAB_LEN = len(tokenizer.word_index)
PE_OUTPUT = MAX_LEN
TARGET_V_SIZE = VOCAB_LEN
REPLICA_BATCH_SIZE = 1

# Config Encoder
PREPROCESSING_FN = tf.keras.applications.efficientnet.preprocess_input
BB_FN = Efficient_Net_encoder.get_efficientnetv2_backbone

# Config Model
testing_config = config.Config()

testing_config.initialize_encoder_config(
    image_embedding_dim=IMG_EMB_DIM,
    preprocessing_fn=PREPROCESSING_FN,
    backbone_fn=BB_FN,
    image_shape=IMG_SHAPE,
    do_permute=IMG_EMB_DIM[1] < IMG_EMB_DIM[0],
)

testing_config.initialize_transformer_config(
    vocab_len=VOCAB_LEN,
    max_len=MAX_LEN,
    n_transformer_layers=N_LAYERS,
    transformer_d_dff=D_FF,
    transformer_n_heads=N_HEADS,
    image_embedding_dim=IMG_EMB_DIM,
)

# print(f"Encoder config:\n\t -> {testing_config.encoder_config}\n")
# print(f"Transformer config:\n\t -> {testing_config.transformer_config}\n")

# Prepare model
optimizer, encoder, transformer = config.prepare_models(
    encoder_config=testing_config.encoder_config,
    transformer_config=testing_config.transformer_config,
    replica_batch_size=REPLICA_BATCH_SIZE,
    verbose=0,
)

# Load trained model checkpoint
checkpoint_path = os.path.join(HERE, "checkpoints_")
ckpt = tf.train.Checkpoint(
    encoder=encoder, transformer=transformer, optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])


def main():
    if len(sys.argv) != 2:
        print("Usage: {} $image_path".format(sys.argv[0]))
    else:
        SMILES = predict_SMILES(sys.argv[1])
        print(SMILES)


def evaluate(image_path: str):
    """
    This function takes an image path (str) and returns the SELFIES
    representation of the depicted molecule (str).

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (str): SELFIES representation of the molecule in the input image
    """
    sample = config.decode_image(image_path)
    _image_batch = tf.expand_dims(sample, 0)
    _image_embedding = encoder(_image_batch, training=False)
    output = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
    result = []
    end_token = tokenizer.word_index["<end>"]

    for i in range(MAX_LEN):
        combined_mask = Transformer_decoder.create_mask(None, output)
        prediction_batch, _ = transformer(
            _image_embedding, output, training=False, look_ahead_mask=combined_mask
        )

        predictions = prediction_batch[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == end_token:
            return result

        result.append(tokenizer.index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return result


def predict_SMILES(image_path: str):
    """
    This function takes an image path (str) and returns the SMILES
    representation of the depicted molecule (str).

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (str): SMILES representation of the molecule in the input image
    """
    predicted_SELFIES = evaluate(image_path)
    predicted_SMILES = decoder(
        "".join(predicted_SELFIES).replace("<start>", "").replace("<end>", "")
    )

    return predicted_SMILES


if __name__ == "__main__":
    main()
