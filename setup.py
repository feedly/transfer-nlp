from setuptools import setup, find_packages
from io import open

setup(
    name='transfer_nlp',
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    version='0.1.6',
    license='MIT',
    description='NLP library designed for flexible research and development',
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Peter Martigny',
    author_email='peter.martigny@gmail.com',
    url='https://github.com/feedly/transfer-nlp',
    download_url='https://github.com/feedly/transfer-nlp/archive/v0.1.6.tar.gz',
    keywords=['NLP', 'transfer learning', 'language models', 'NLU'],
    install_requires=[
        'numpy>=1.16.2',
        'pyaml>=19.4.1',
        'toml>=0.10.0'
    ],
    extras_require={
        'torch': [
            'torch>=1.1.0',
            'pytorch-ignite>=0.2.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
