Transfer NLP Documentation
==========================

:mod:`Transfer NLP` is a framework built on top of PyTorch which goal is to achieve 2 kinds of Transfer:

- **easy transfer of code**: the framework should be modular enough so that you don't have to re-write everything each time you experiment with a new architecture / a new kind of task
- **easy transfer learning**: the framework should be able to easily interact with pre-trained models and manipulate them in order to fine-tune some of their parts.

You can try the library on this `Colab Notebook <https://colab.research.google.com/drive/1DtC31eUejz1T0DsaEfHq_DOxEfanmrG1#scrollTo=Xzu3HPdGrnza>`_., which shows how to use the framework on several examples.
All examples on these notebooks embed in-cell Tensorboard training monitoring!

Installation
============
From source:

You can clone the source from `github <https://github.com/feedly/transfer-nlp>`_ and run

.. code:: bash

    python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Notes

   concepts
   data-components
   modeling-components
   trainer-components
   faq


.. toctree::
   :maxdepth: 2
   :caption: Data Management

   vocabulary
   vectorizer
   loader

.. toctree::
   :maxdepth: 2
   :caption: Experiment Management

   config


.. toctree::
   :maxdepth: 2
   :caption: Package Components

   trainers
   predictors
   regularizers


.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   license
   help


