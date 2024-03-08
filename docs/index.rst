.. DECIMER documentation master file, created by
   Kohulan Rajan on Thu Feb 17 16:47:38 2022.

Welcome to DECIMER's documentation!
=====================================

.. image:: https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_V2.png?raw=true
  :width: 500
  :align: center

The DECIMER 2.0 (Deep lEarning for Chemical ImagE Recognition) project was launched to address the OCSR problem with the latest computational intelligence methods to provide an automated open-source software solution.

The original implementation of DECIMER using GPU takes a longer training time when we use a bigger dataset of more than 1 million images. To overcome these longer training times, many implement the training script to work on multiple GPUs. However, we tried to step up and implemented our code to use Google's Machine Learning hardware TPU(Tensor Processing Unit).

For comments, bug reports or feature ideas, please use github issues
or send an email to kohulan.rajan@uni-jena.de

Installation
============

Install DECIMER in the command line using pip:

.. code-block::

   pip install decimer

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
