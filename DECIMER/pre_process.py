import io
import cv2
import efficientnet.tfkeras as efn
import tensorflow as tf
import numpy as np

from PIL import Image
from PIL import ImageEnhance
from pillow_heif import register_heif_opener
from typing import Union


def resize_byratio(image):
    """This function takes a Pillow Image object and will resize the image by
    512 x 512 To upscale or to downscale the image LANCZOS resampling method is
    used.

    with the new pillow version the antialias is turned on when using LANCZOS.
    Args: PIL.Image
    Returns: PIL.Image
    """
    maxwidth = 512
    ratio = maxwidth / max(image.width, image.height)
    new_size = int((float(image.width) * ratio)), int((float(image.height) * ratio))
    resized_image = image.resize(new_size, resample=Image.LANCZOS)
    return resized_image


def central_square_image(image):
    """This function takes a Pillow Image object and will add white padding so
    that the image has a square shape with the width/height of the longest side
    of the original image.

    Args: PIL.Image
    Returns: PIL.Image
    """
    max_wh = int(1.2 * max(image.size))
    # If the new image is smaller than 299x299, then let's paste it into an empty image
    # of that size instead of distorting it later while resizing.
    if max_wh < 512:
        max_wh = 512
    new_im = Image.new(image.mode, (max_wh, max_wh), "white")
    paste_pos = (
        int((new_im.size[0] - image.size[0]) / 2),
        int((new_im.size[1] - image.size[1]) / 2),
    )
    new_im.paste(image, paste_pos)
    return new_im


def delete_empty_borders(image):
    """This function takes a Pillow Image object, converts it to grayscale and
    deletes white space at the borders.

    Args: PIL.Image
    Returns: PIL.Image
    """
    image = np.asarray(image.convert("L"))
    mask = image > 200
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    crop = image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return Image.fromarray(crop)


def PIL_im_to_BytesIO(image):
    """
    Convert pillow image to io.BytesIO object
    Args: PIL.Image
    Returns: io.BytesIO object with the image data
    """
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output


def HEIF_to_pillow(image_path: str):
    """
    Converts Apple's HEIF format to useful pillow object
    Returns: image_path (str): path of input image
    Returns: PIL.Image
    """
    register_heif_opener()

    heif_file = Image.open(image_path).convert("RGBA")
    return heif_file


def remove_transparent(image: Union[str, np.ndarray]) -> Image.Image:
    """
    Removes the transparent layer from a PNG image with an alpha channel.

    Args:
        image (Union[str, np.ndarray]): Path of the input image or a numpy array representing the image.

    Returns:
        PIL.Image.Image: The image with transparency removed.
    """

    def process_image(png: Image.Image) -> Image.Image:
        """
        Helper function to remove transparency from a single image.

        Args:
            png (PIL.Image.Image): The input PIL image with transparency.

        Returns:
            PIL.Image.Image: The image with transparency removed.
        """
        background = Image.new("RGBA", png.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, png)
        return alpha_composite

    def handle_image_path(image_path: str) -> Image.Image:
        """
        Helper function to handle image paths.

        Args:
            image_path (str): The path to the input image.

        Returns:
            PIL.Image.Image: The image with transparency removed.
        """
        try:
            png = Image.open(image_path).convert("RGBA")
        except Exception as e:
            if type(e).__name__ == "UnidentifiedImageError":
                png = HEIF_to_pillow(image_path)
            else:
                print(e)
                raise Exception
        return process_image(png)

    def handle_numpy_array(array: np.ndarray) -> Image.Image:
        """
        Helper function to handle a numpy array.

        Args:
            array (np.ndarray): The numpy array representing the image.

        Returns:
            PIL.Image.Image: The image with transparency removed.
        """
        png = Image.fromarray(array).convert("RGBA")
        return process_image(png)

    # Check if input is a numpy array
    if isinstance(image, np.ndarray):
        return handle_numpy_array(array=image)

    return handle_image_path(image_path=image)


def get_bnw_image(image):
    """
    converts images to black and white
    Args: PIL.Image
    Returns: PIL.Image
    """

    im_np = np.asarray(image)
    grayscale = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    # (thresh, im_bw) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_pil = Image.fromarray(grayscale)
    enhancer = ImageEnhance.Contrast(im_pil)
    im_output = enhancer.enhance(1.8)
    return im_output


def increase_contrast(image):
    """This function increases the contrast of an image input.

    Args: PIL.Image
    Returns: PIL.Image
    """

    # Get brightness range - cast to int to avoid uint8 overflow
    min_val = int(np.min(image))
    max_val = int(np.max(image))

    # Use LUT to convert image values
    LUT = np.zeros(256, dtype=np.uint8)
    LUT[min_val : max_val + 1] = np.linspace(
        start=0, stop=255, num=(max_val - min_val) + 1, endpoint=True, dtype=np.uint8
    )

    # Apply LUT and return image
    return Image.fromarray(LUT[image])


def get_resize(image):
    """This function used to decide how to resize a given image without losing
    much information.

    Args: PIL.Image
    Returns: PIL.Image
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


def increase_brightness(image):
    """This function adjusts the brightness of the given image.

    Args: PIL.Image
    Returns: PIL.Image
    """
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.6)
    return image


def decode_image(image_path: Union[str, np.ndarray]):
    """Loads an image and preprocesses the input image in several steps to get
    the image ready for DECIMER input.

    Args:
        image_path (Union[str, np.ndarray]): path of input image or numpy array representing the image.

    Returns:
        Processed image
    """
    img = remove_transparent(image_path)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    img = PIL_im_to_BytesIO(img)
    img = tf.image.decode_png(img.getvalue(), channels=3)
    img = tf.image.resize(img, (512, 512), method="gaussian", antialias=True)
    img = efn.preprocess_input(img)
    return img
