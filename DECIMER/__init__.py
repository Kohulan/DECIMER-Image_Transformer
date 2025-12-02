# -*- coding: utf-8 -*-
"""
DECIMER V2.7.2 Python Package
=============================

Deep lEarning for Chemical ImagE Recognition (DECIMER) project
was launched to address the Optical Chemical Structure Recognition (OCSR)
problem using deep learning based methods,
providing an automated open-source software solution.

Typical usage example::

    from decimer import predict_SMILES

    # Chemical depiction to SMILES translation
    image_path = "path/to/imagefile"
    SMILES = predict_SMILES(image_path)
    print(SMILES)

For comments, bug reports, or feature ideas,
please raise an issue on the GitHub repository.
"""

__version__ = "2.7.2"

__all__ = [
    "DECIMER",
]


from .decimer import predict_SMILES
