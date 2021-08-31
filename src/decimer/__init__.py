# -*- coding: utf-8 -*-

"""DECIMER Python Package.

DECIMER : Deep lEarning for Chemical ImagE Recognition

Translates an image with a chemical structure depiction to SMILES.

The DECIMER 1.0 (Deep lEarning for Chemical ImagE Recognition) project was launched to address the OCSR problem with the latest computational intelligence methods to provide an automated open-source software solution. Also in future, this can be completely automatable.


For easy usage refer to the decimer command-line python script.
Note: The default model is set to predict Canonical SMILES.

	Available models:
	================

  - Canonical: Model trained on images depicted using canonical SMILES (without stereochemistry).
  - Isomeric: Model trained on images depicted using isomeric SMILES (with stereochemistry).
  - Augmented: Model trained on images depicted using isomeric SMILES with augmentations.

"""
