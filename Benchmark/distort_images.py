import os
import sys
from multiprocessing import Pool

import imgaug.augmenters as iaa
import numpy as np
from skimage.io import imread
from skimage.io import imsave


def distort_image(input_image_path: str, output_image_path: str) -> None:
    """This function takes the path of an input image and the desired output
    image path (both str). It rotates the input image randomly (-5° or +5°) and
    applies shearing (in x or y-direction with an angle of -9° or 9°) as
    described by Clevert et al (Img2Mol (preprint)). I was a bit confused about
    the "shearing factor" of +/- 0.1 described by Clevert et al. Normally the
    transformation is done with a shearing angle. I assume here that the
    correponding angle is 0.1*90° = 9° as 90° is the extreme case where the
    image turns into a horizontal or vertical line. This seems reasonable as it
    results in a mild distortion as described by the authors.

    Args:
            input_image_path (str): Path of input image
            output_image_path (str): Path where the output image is saved
    """
    image = imread(input_image_path)
    # Add a one-pixel-layer of white pixels so that we can simply take that as a reference what to fill
    # the blank space with when we apply the transformations
    image = np.pad(image, pad_width=1, mode="constant", constant_values=255)
    augmentation = iaa.Sequential(
        [
            iaa.Affine(rotate=[-5, 5], mode="edge", fit_output=True),
            iaa.OneOf(
                [
                    iaa.geometric.ShearX([-9, 9], mode="edge", fit_output=True),
                    iaa.geometric.ShearY([-9, 9], mode="edge", fit_output=True),
                ]
            ),
        ]
    )
    image = augmentation(images=[image])[0]
    imsave(fname=output_image_path, arr=image)
    return


def main():
    """This script takes an input directory with images of chemical structure
    depictions, applies a random rotation (-5° or +5°) as well as shearing
    (angle drawn from [-0.1, 0.1]) to every input image.

    These distortions are supposed to imitate the image modifications
    described by Clevert et al (Img2Mol (preprint))
    """
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    starmap_iterable = [
        (os.path.join(input_dir, image_name), os.path.join(output_dir, image_name))
        for image_name in os.listdir(input_dir)
        if image_name[-3:].lower() == "png"
    ]
    with Pool(5) as p:
        _ = p.starmap(distort_image, starmap_iterable)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} input_dir output_dir".format(sys.argv[0]))
    else:
        main()
