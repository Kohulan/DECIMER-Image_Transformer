import os
import pickle
import sys
from typing import List
from typing import Tuple

import pystow
import tensorflow as tf
import Transformer_decoder

if int(tf.__version__.split(".")[1]) <= 10:
    import Efficient_Net_encoder
else:
    raise ImportError("Please use tensorflow 2.10 when working with the checkpoints.")
import config
import utils

print(tf.__version__)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
"""
# Set path
default_path = pystow.join("DECIMER-V2")

# model download location
checkpoint_url = "https://zenodo.org/record/8093783/files/DECIMER_512_checkpoints.zip"
checkpoint_path = str(default_path) + "/DECIMER_checkpoints/"

if not os.path.exists(checkpoint_path):
    config.download_trained_weights(checkpoint_url, default_path)

# load assets

tokenizer = pickle.load(
    open(
        default_path.as_posix() + "/DECIMER_model/assets/tokenizer_SMILES.pkl",
        "rb",
    )
)

max_length = 302

# Image parameters
IMG_EMB_DIM = (16, 16, 512)
IMG_EMB_DIM = (IMG_EMB_DIM[0] * IMG_EMB_DIM[1], IMG_EMB_DIM[2])
IMG_SHAPE = (512, 512, 3)
PE_INPUT = IMG_EMB_DIM[0]
IMG_SEQ_LEN, IMG_EMB_DEPTH = IMG_EMB_DIM
D_MODEL = 512

# Network parameters
N_LAYERS = 6
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
    image_embedding_dim=D_MODEL,
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
ckpt = tf.train.Checkpoint(
    encoder=encoder, transformer=transformer, optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])


def main():
    if len(sys.argv) != 2:
        print("Usage: {} $image_path".format(sys.argv[0]))
    else:
        SMILES = predict_SMILES(sys.argv[1])
        print(SMILES)
        SMILES_with_confidence = predict_SMILES_with_confidence(sys.argv[1])
        for tup in SMILES_with_confidence:
            print(tup)


class DECIMER_Predictor(tf.Module):
    def __init__(self, encoder, tokenizer, transformer, max_length):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_length = max_length

    def __call__(self, Decoded_image):
        """Run the DECIMER predictor model when called. Usage of predict_SMILES
        or predict_SMILES_with_confidence is recommended instead.

        Args:
            Decoded_image (_type_): output of config.decode_image

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: predicted tokens, confidence values
        """
        assert isinstance(Decoded_image, tf.Tensor)

        _image_batch = tf.expand_dims(Decoded_image, 0)
        _image_embedding = encoder(_image_batch, training=False)

        start_token = tf.cast(
            tf.convert_to_tensor([tokenizer.word_index["<start>"]]), tf.int32
        )
        end_token = tf.cast(
            tf.convert_to_tensor([tokenizer.word_index["<end>"]]), tf.int32
        )

        output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start_token)
        confidence_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for t in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            combined_mask = Transformer_decoder.create_masks_decoder(output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            prediction_batch = transformer(
                output, _image_embedding, training=False, look_ahead_mask=combined_mask
            )

            # select the last word from the seq_len dimension
            predictions = prediction_batch[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            confidence = predictions[-1][-1][int(predicted_id)]
            output_array = output_array.write(t + 1, predicted_id[0])
            confidence_array = confidence_array.write(t + 1, confidence)
            if predicted_id == end_token:
                break
        output = tf.transpose(output_array.stack())

        return output, confidence_array.stack()


def detokenize_output(predicted_array: tf.Tensor) -> str:
    """This function takes the predicted array of tokens and returns the
    predicted SMILES string.

    Args:
        predicted_array (tf.Tensor): Transformer Decoder output array (predicted tokens)

    Returns:
        str: SMILES string
    """
    outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        "".join([str(elem) for elem in outputs])
        .replace("<start>", "")
        .replace("<end>", "")
    )
    return prediction


def detokenize_output_add_confidence(
    predicted_array: tf.Tensor,
    confidence_array: tf.Tensor,
) -> List[Tuple[str, float]]:
    """This function takes the predicted array of tokens as well as the
    confidence values returned by the Transformer Decoder and returns a list of
    tuples that contain each token of the predicted SMILES string and the
    confidence value.

    Args:
        predicted_array (tf.Tensor): Transformer Decoder output array (predicted tokens)

    Returns:
        str: SMILES string
    """
    prediction_with_confidence = [
        (
            tokenizer.index_word[predicted_array[0].numpy()[i]],
            confidence_array[i].numpy(),
        )
        for i in range(len(confidence_array))
    ]
    decoded_prediction_with_confidence = list(
        [(utils.decoder(tok), conf) for tok, conf in prediction_with_confidence[1:-1]]
    )
    decoded_prediction_with_confidence.append(prediction_with_confidence[-1])
    return decoded_prediction_with_confidence


# Initiate the DECIMER class
DECIMER = DECIMER_Predictor(encoder, tokenizer, transformer, MAX_LEN)


def predict_SMILES_with_confidence(image_path: str) -> List[Tuple[str, float]]:
    """This function takes an image path (str) and returns a list of tuples
    that contain each token of the predicted SMILES string and the confidence
    level from the last layer of the Transformer decoder.

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (List[Tuple[str, float]]): Tuples that contain the tokens and the confidence
            values of the predicted SMILES
    """
    decodedImage = config.decode_image(image_path)
    predicted_tokens, confidence_values = DECIMER(tf.constant(decodedImage))
    predicted_SMILES_with_confidence = detokenize_output_add_confidence(
        predicted_tokens, confidence_values
    )
    return predicted_SMILES_with_confidence


def predict_SMILES(image_path: str) -> str:
    """This function takes an image path (str) and returns the SMILES
    representation of the depicted molecule (str).

    Args:
        image_path (str): Path of chemical structure depiction image

    Returns:
        (str): SMILES representation of the molecule in the input image
    """
    decodedImage = config.decode_image(image_path)
    predicted_tokens, _ = DECIMER(tf.constant(decodedImage))
    predicted_SMILES = utils.decoder(detokenize_output(predicted_tokens))
    return predicted_SMILES


if __name__ == "__main__":
    main()
