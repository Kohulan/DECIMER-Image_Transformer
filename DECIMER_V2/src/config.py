# Network configuration file
import tensorflow as tf
import efficientnet.tfkeras as efn
import Efficient_Net_encoder
import Transformer_decoder
from PIL import Image
import numpy as np
import io


TARGET_DTYPE = tf.float32

def central_square_image(im):
    """
    This function takes a Pillow Image object and will add white padding 
    so that the image has a square shape with the width/height of the longest side 
    of the original image.
    ___
    im: PIL.Image
    ___
    output: PIL.Image
    """
    max_wh = int(1.2 * max(im.size))
    # If the new image is smaller than 299x299, then let's paste it into an empty image
    # of that size instead of distorting it later while resizing.
    if max_wh < 299:
        max_wh = 299
    new_im = Image.new(im.mode, (max_wh, max_wh), "white")
    paste_pos = (int((new_im.size[0]-im.size[0])/2), int((new_im.size[1]-im.size[1])/2))
    new_im.paste(im, paste_pos)
    return new_im


def delete_empty_borders(im):
    """This function takes a Pillow Image object, converts it to grayscale and
    deletes white space at the borders.
    ___
    im: PIL.Image
    ___
    output: PIL.Image
    """
    im = np.asarray(im.convert('L'))
    mask = im > 200
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    crop = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return Image.fromarray(crop)


def PIL_im_to_BytesIO(im):
    """    
    Convert pillow image to io.BytesIO object
    ___
    im: PIL.Image
    ___
    Output: io.BytesIO object with the image data
    """
    output = io.BytesIO()
    im.save(output, format='PNG')
    return output


def decode_image(image_path: str):
    """
    Loads and preprocesses an image
    Args:
        image_path (str): path of input image

    Returns:
        Processed image
    """
    img = Image.open(image_path)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = PIL_im_to_BytesIO(img)
    img = tf.image.decode_png(img.getvalue(), channels=3)
    img = tf.image.resize(img, (299, 299))
    img = efn.preprocess_input(img)
    return img


class Config:
    """
    Configuration class
    """
    def __init__(
        self,
    ):
        self.encoder_config = {}
        self.transformer_config = {}
        self.lr_config = {}

    def initialize_encoder_config(
        self,
        image_embedding_dim,
        preprocessing_fn,
        backbone_fn,
        image_shape,
        do_permute=False,
        pretrained_weights=None,
    ):
        self.encoder_config = dict(
            image_embedding_dim=image_embedding_dim,
            preprocessing_fn=preprocessing_fn,
            backbone_fn=backbone_fn,
            image_shape=image_shape,
            do_permute=do_permute,
            pretrained_weights=pretrained_weights,
        )

    def initialize_transformer_config(
        self,
        vocab_len,
        max_len,
        n_transformer_layers,
        transformer_d_dff,
        transformer_n_heads,
        image_embedding_dim,
        dropout_rate=0.1,
    ):
        self.transformer_config = dict(
            num_layers=n_transformer_layers,
            d_model=image_embedding_dim[-1],
            num_heads=transformer_n_heads,
            dff=transformer_d_dff,
            target_vocab_size=vocab_len,
            pe_input=image_embedding_dim[0],
            pe_target=max_len,
            dropout_rate=0.1,
        )

    def initialize_lr_config(self, warm_steps, n_epochs):
        self.lr_config = dict(
            warm_steps=warm_steps,
            n_epochs=n_epochs,
        )


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def prepare_models(encoder_config, transformer_config, replica_batch_size, verbose=0):

    # Instiate an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)

    # Instantiate the encoder model

    encoder = Efficient_Net_encoder.Encoder(**encoder_config)
    initialization_batch = encoder(
        tf.ones(
            ((replica_batch_size,) + encoder_config["image_shape"]), dtype=TARGET_DTYPE
        ),
        training=False,
    )

    # Instantiate the decoder model
    transformer = Transformer_decoder.Transformer(**transformer_config)
    transformer(
        initialization_batch, tf.random.uniform((replica_batch_size, 1)), training=False
    )

    # Show the model architectures and plot the learning rate
    if verbose:
        print("\nEncoder model\n")
        print(encoder.summary())

        print("\nTransformer model\n")
        print(transformer.summary())

    return optimizer, encoder, transformer
