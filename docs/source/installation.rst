============
Installation
============


Prerequisites
-------------

ugtm requires Python >= 3.8 and the following packages:

  - numpy >= 1.21
  - scikit-learn >= 1.0
  - scipy >= 1.7
  - jinja2 >= 3.0


pip installation
----------------

Install using pip::

        pip install ugtm

We recommend using a virtual environment::

        python -m venv .venv
        source .venv/bin/activate   # on Windows: .venv\Scripts\activate
        pip install ugtm


Using conda
-----------

Create a conda environment and install via pip::

        conda create -n ugtm python=3.11
        conda activate ugtm
        pip install ugtm


Import package
--------------

In a Python console::

        import ugtm
