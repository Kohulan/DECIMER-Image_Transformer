# Network configuration file
import tensorflow as tf
import efficientnet.tfkeras as efn
import DECIMER.Efficient_Net_encoder
import DECIMER.Transformer_decoder
from PIL import Image, ImageEnhance
import numpy as np
import io
import cv2
import pystow
import pathlib
import pyheif
import zipfile

TARGET_DTYPE = tf.float32


def resize_byratio(image):
    """
    This function takes a Pillow Image object and will resize the image by 512 x 512
    To upscale or to downscale the image LANCZOS resampling method is used.
    with the new pillow version the antialias is turned on when using LANCZOS.
    ___
    im: PIL.Image
    ___
    output: PIL.Image
    """
    maxwidth = 512
    ratio = maxwidth / max(image.width, image.height)
    new_size = int((float(image.width) * ratio)), int((float(image.height) * ratio))
    resized_image = image.resize(new_size, resample=Image.LANCZOS)
    return resized_image


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
    if max_wh < 512:
        max_wh = 512
    new_im = Image.new(im.mode, (max_wh, max_wh), "white")
    paste_pos = (
        int((new_im.size[0] - im.size[0]) / 2),
        int((new_im.size[1] - im.size[1]) / 2),
    )
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
    im = np.asarray(im.convert("L"))
    mask = im > 200
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    crop = im[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
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
    im.save(output, format="PNG")
    return output


def HEIF_to_pillow(image_path: str):
    """
    Converts Apple's HEIF format to useful pillow object
    ___
    image_path (str): path of input image
    ___
    Output: PIL.Image
    """
    heif_file = pyheif.read(image_path)
    pil_im = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return pil_im.convert("RGBA")


def remove_transparent(image_path: str):
    """
    Removes the transparent layer from a PNG image with an alpha channel
    ___
    image_path (str): path of input image
    ___
    Output: PIL.Image
    """
    if pathlib.Path(image_path).suffix == ".HEIC":
        png = HEIF_to_pillow(image_path)
    else:
        png = Image.open(image_path).convert("RGBA")

    background = Image.new("RGBA", png.size, (255, 255, 255))

    alpha_composite = Image.alpha_composite(background, png)

    return alpha_composite


def get_bnw_image(image):
    """
    converts images to black and white
    ___
    image: PIL.Image
    ___
    Output: PIL.Image
    """

    im_np = np.asarray(image)
    grayscale = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    # (thresh, im_bw) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_pil = Image.fromarray(grayscale)
    enhancer = ImageEnhance.Contrast(im_pil)
    im_output = enhancer.enhance(1.2)
    return im_output


def increase_contrast(image):
    """
    This function increases the contrast of an image input.
    ___
    image: PIL.Image
    ___
    Output: PIL.Image
    """

    # Get brightness range
    min = np.min(image)
    max = np.max(image)

    # Use LUT to convert image values
    LUT = np.zeros(256, dtype=np.uint8)
    LUT[min : max + 1] = np.linspace(
        start=0, stop=255, num=(max - min) + 1, endpoint=True, dtype=np.uint8
    )

    # Apply LUT and return image
    return Image.fromarray(LUT[image])


def get_resize(image):
    """
    This function used to decide how to resize a given image without losing much information.
    ___
    image: PIL.Image
    ___
    Output: PIL.Image
    """

    width, height = image.size

    # Checks the image size and resizes using the LANCOS image resampling
    if (width == height) and (width < 512):
        image = image.resize((512, 512), resample=Image.LANCZOS)

    elif width >= 512 or height >= 512:
        return image

    else:
        image = resize_byratio(image)

    return image


def decode_image(image_path: str):
    """
    Loads an image and preprocesses the input image in several steps to get the image ready for DECIMER input.

    Args:
        image_path (str): path of input image

    Returns:
        Processed image
    """
    img = remove_transparent(image_path)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = PIL_im_to_BytesIO(img)
    img = tf.image.decode_png(img.getvalue(), channels=3)
    img = tf.image.resize(img, (512, 512), method="gaussian", antialias=True)
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
        """This functions initializes the Efficient-Net V2 encoder with user defined
        configurations.

        Args:
            image_embedding_dim (int): Embedding dimention of the input image
            preprocessing_fn (method): Efficient Net preprocessing function for input image
            backbone_fn (method): Calls Efficient-Net V2 as backbone for encoder
            image_shape (int): Shape of the input image
            do_permute (bool, optional): . Defaults to False.
            pretrained_weights (keras weights, optional): Use pretrainined efficient net weights or not. Defaults to None.
        """
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
        """This functions initializes the Transformer model as decoder with user defined
        configurations.


        Args:
            vocab_len (int): Total number of words in the input vocabulary
            max_len (int): Maximum length of the string found on the training dataset
            n_transformer_layers (int): Number of layers present in the transformer model
            transformer_d_dff (int): Transformer feed forward upwards projection size
            transformer_n_heads (int): Number of heads present in the transformer model
            image_embedding_dim (int): Total number of dimension the image gets embeddeded
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.1.
        """
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
        """This function sets the configuration to initialize learning rate

        Args:
            warm_steps (int): Number of steps The learning rate is increased
            n_epochs (int): Number of epochs
        """
        self.lr_config = dict(
            warm_steps=warm_steps,
            n_epochs=n_epochs,
        )


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom schedule for learning rate used during training

    Args:
        tf (_type_): keras learning rate schedule
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def prepare_models(encoder_config, transformer_config, replica_batch_size, verbose=0):
    """This function is used to initiate the Encoder and the Transformer with appropriate
    configs set by the user. After initiating the models this function returns the Encoder,Transformer
    and the optimizer.

    Args:
        encoder_config ([type]): Encoder configuration set by user in the config class.
        transformer_config ([type]): Transformer configuration set by user in the config class.
        replica_batch_size ([type]): Per replica batch size set by user(during distributed training).
        verbose (int, optional): Defaults to 0.

    Returns:
        [type]: Optimizer, Encoder model and the Transformer
    """

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


# Downloads the model and unzips the file downloaded, if the model is not present on the working directory.
def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained models and tokenizers to a default location.
    After downloading the zipped file the function unzips the file automatically.
    If the model exists on the default location this function will not work.
    Args:
        model_url (str): trained model url for downloading.
        model_path (str): model default path to download.
    Returns:
        downloaded model.
    """
    # Download trained models
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
        model_path = pystow.ensure("DECIMER-V2", url=model_url)
        print(model_path)
    if verbose > 0:
        print("... done downloading trained model!")
        with zipfile.ZipFile(model_path.as_posix(), "r") as zip_ref:
            zip_ref.extractall(model_path.parent.as_posix())
