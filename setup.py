from distutils.core import setup
from io import open


setup(
    name='transfer_nlp',
    packages=['transfer_nlp'],
    version='0.0.2',
    license='MIT',
    description='NLP library designed for flexible research and development',
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author='Peter Martigny',
    author_email='peter.martigny@gmail.com',
    url='https://github.com/feedly/transfer-nlp',
    download_url='https://github.com/feedly/transfer-nlp/archive/V0.0.2.zip',
    keywords=['NLP', 'transfer learning', 'language models', 'NLU'],
    install_requires=[
        'tqdm==4.31.1',
        'matplotlib==3.0.3',
        'nltk==3.4',
        'pandas==0.24.1',
        'seaborn==0.9.0',
        'scipy==1.2.1',
        'annoy==1.15.1',
        'numpy==1.16.2',
        'requests==2.21.0',
        'torch==1.0.1.post2',
        'ipython==6.2.1',
        'feedly-client==0.19',
        'bs4==0.0.1',
        'tensorboardX==1.6',
        'knockknock==0.1'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
