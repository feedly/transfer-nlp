import logging
import re
from typing import List

name = 'transfer_nlp.common.tokenizers'
logging.getLogger(name).setLevel(level=logging.INFO)
logger = logging.getLogger(name)


def tokenize(text: str) -> List[str]:
    """
    Basic text preprocessing
    :param text:
    :return:
    """

    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    tokens = text.split(" ")
    if not tokens[-1]:
        tokens = tokens[:-1]

    return tokens


if __name__ == "__main__":
    logging.info('')

    example = "Hello world!"
    tokenized = tokenize(text=example)
    logger.info(f"{tokenized}")
