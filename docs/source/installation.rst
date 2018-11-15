============
Installation
============


Prerequisites
--------------

ugtm requires Python 2.7 or + (tested on Python 3.4.6 and Python 2.7.14),
with following packages:

  - scikit-learn>=0.20
  - numpy>=1.14.5
  - matplotlib>=2.2.2
  - scipy>=0.19.1
  - mpld3>=0.3
  - jinja2>=2.10


pip installation
----------------

Install using pip in the command line::

        pip install ugtm

If this does not work, try upgrading packages::

        sudo pip install --upgrade pip numpy scikit-learn matplotlib scipy mpld3 jinja2


Using anaconda
--------------

Example of anaconda virtual env "p2" for python 2.7.14::

        conda create -n p2 python=2.7.14 numpy=1.14.5 \
        scikit-learn=0.20 matplotlib=2.2.2 \
        scipy=0.19.1 mpld3=0.3 jinja2=2.10

        # Activate virtual env  
        source activate p2

        # Install package
        pip install ugtm

Example of anaconda virtual env p3 for python 3.6.6::

        conda create -n p3 python=3.6.6 numpy=1.14.5 \
        scikit-learn=0.20 matplotlib=2.2.2 \
        scipy=0.19.1 mpld3=0.3 jinja2=2.10

        # Activate virtual env
        source activate p3

        # Install package
        pip install ugtm


Import package
--------------

In python console, import ugtm package::

        import ugtm


