import os
import tensorflow as tf

import pickle
import Transformer_decoder
import Efficient_Net_encoder
import config

print(tf.__version__)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load assets
tokenizer = pickle.load(open("tokenizer_TPU_Stereo.pkl", "rb"))
max_length = pickle.load(open("max_length_TPU_Stereo.pkl", "rb"))

# Image parameters
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

# Prepare model
optimizer, encoder, transformer = config.prepare_models(
    encoder_config=testing_config.encoder_config,
    transformer_config=testing_config.transformer_config,
    replica_batch_size=REPLICA_BATCH_SIZE,
    verbose=0,
)

# Load trained model checkpoint
checkpoint_path = "checkpoints"
ckpt = tf.train.Checkpoint(
    encoder=encoder, transformer=transformer, optimizer=optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
    start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])


class DECIMER_Predictor(tf.Module):
    """This is a class which takes care of inference. It loads the saved checkpoint and the necessary
    tokenizers. The inference begins with the start token (<start>) and ends when the end token(<end>)
    is met. This class can only work with tf.Tensor objects. The strings shoul gets transformed into np.arrays
    before feeding them into this class.
    """

    def __init__(self, encoder, tokenizer, transformer, max_length):
        """Load the tokenizers, the maximum input and output length and the model.

        Args:
            encoder (tf.keras.model):  The encoder model
            tokenizer (tf.keras.tokenizer): Output tokenizer, defines which charater is assigned to what token
            transformer (tf.keras.model):  The transformer model
            max_length (int): Maximum length of a string which can get predicted
        """
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_length = max_length

    def __call__(self, Decoded_image):
        """This fuction takes in the Decoded image as input and
        makes the predicted list of tokens and return the tokens as tf.Tensor array.
        Before feeding the input array we must define start and the end tokens.

        Args:
            Decoded_image (tf.Tensor[tf.int32]): Input array in tf.Eagertensor format.

        Returns:
            output (tf.Tensor[tf.int64]): predicted output as an array.
        """

        assert isinstance(Decoded_image, tf.Tensor)
        if len(Decoded_image.shape) == 0:
            Decoded_image = Decoded_image[tf.newaxis]

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

        for t in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            combined_mask = Transformer_decoder.create_mask(None, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            prediction_batch, _ = transformer(
                _image_embedding, output, training=False, look_ahead_mask=combined_mask
            )

            # select the last word from the seq_len dimension
            predictions = prediction_batch[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output_array = output_array.write(t + 1, predicted_id[0])

            if predicted_id == end_token:
                break
        output = tf.transpose(output_array.stack())

        return output


DECIMER = DECIMER_Predictor(encoder, tokenizer, transformer, MAX_LEN)


class ExportDECIMERPredictor(tf.Module):
    """This class wraps the inference class into a module into tf.Module sub-class, with a tf.function on the __call__ method.
    So we could export the model as a tf.saved_model.
    """

    def __init__(self, DECIMER):
        """Import the translator instance."""
        self.DECIMER = DECIMER

    @tf.function
    def __call__(self, Decoded_Image):
        """This fucntion calls the __call__function from the translator class.
        In the tf.function only the output sentence is returned.
        Thanks to the non-strict execution in tf.function any unnecessary values are never computed.

        Args:
            sentence (tf.Tensor[tf.int32]): Input array in tf.Easgertensor format.

        Returns:
            tf.Tensor[tf.int64]: predicted output as an array.
        """

        result = self.DECIMER(Decoded_Image)

        return result


DECIMER_export = ExportDECIMERPredictor(DECIMER)

tf.saved_model.save(
    DECIMER_export,
    export_dir="DECIMER_Packed_model",
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
)
