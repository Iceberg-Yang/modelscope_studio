"""Triton kernels bundled with SageAttention.

This marker file is required because SageAttention's setup.py uses
``find_packages()``. Without it, the ``sageattention.triton`` modules are
omitted from the built wheel and importing ``sageattention`` fails before any
CUDA extension can be loaded.
"""
