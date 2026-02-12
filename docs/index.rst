text2glioma documentation
=========================

**Text- and mask-conditioned 3D latent diffusion for synthetic multi-sequence glioma MRI generation.**

text2glioma is a Python package that generates realistic synthetic 3D brain MRI
volumes across four sequences (T1, T1CE, T2, FLAIR) from free-text radiology
prompts and/or tumour segmentation masks.  It uses a two-stage latent diffusion
pipeline: a 3D variational autoencoder (VAE) compresses multi-channel MRI into
a compact latent space, and a text+mask-conditioned diffusion model synthesises
new latent representations that decode into full-resolution 4-channel NIfTI
volumes.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   tutorial
   architecture
   configuration
   validation_plan
   validation_tutorial
   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
