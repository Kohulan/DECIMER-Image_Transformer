# -*- coding: utf-8 -*-
"""DECIMER V2.6.0 Python Package. ============================

This repository contains DECIMER-V2,Deep lEarning for Chemical ImagE Recognition) project
was launched to address the OCSR problem with the latest computational intelligence methods
to provide an automated open-source software solution.


Typical usage example:

from decimer import predict_SMILES

# Chemical depiction to SMILES translation
image_path = "path/to/imagefile"
SMILES = predict_SMILES(image_path)
print(SMILES)

For comments, bug reports or feature ideas,
please raise a issue on the Github repository.
"""

__version__ = "2.6.0"

__all__ = [
    "DECIMER",
]


from .decimer import predict_SMILES
