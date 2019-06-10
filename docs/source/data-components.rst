Data Management Components
==========================


Vocabularies
------------
We provide classes to build vocabularies over datasets.
These classes do not take into account the nature of the symbols which which you are filling a dictionary.
Hence, whether you want to use vocabularies for tokens, characters, BPE, etc.., you can still use the vocabulary classes
coupled with a vectorizer of your choice.


Vectorizers
-----------
Vectorizers take string inputs and converts hem to lists of symbols.
When implementing your vectorizer, you need to build the vocabularies that you need for your experiment, and set these
vocabularies as vectorizer attributes. You also need to implement the `vectorize` method, which turns a string input
into a list of numbers representing the symbols you choose to use to represent the text.

Loaders
-------
Data Loaders splits the dataset into train, validation and test sets, and creates the appropriate PyTorch DataLoaders.


