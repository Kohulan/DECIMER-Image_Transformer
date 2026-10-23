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

Importing DECIMER does not download or load any model. The models are downloaded
(if they are missing) and loaded when predict_SMILES() is first called, or earlier
by calling ``load_models()``. They are stored in ``$PYSTOW_HOME/DECIMER-V2/``
(``~/.data/DECIMER-V2/`` by default); a directory that already holds them is used
without any network access.

For comments, bug reports, or feature ideas,
please raise an issue on the GitHub repository.
"""

__version__ = "2.7.2"

__all__ = [
    "DECIMER",
]


from .decimer import load_models
from .decimer import predict_SMILES
