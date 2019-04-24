import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class TokenizerABC:

    def __init__(self):
        pass

    def tokenize(self, text: str):
        raise NotImplementedError


class CustomTokenizer(TokenizerABC):

    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
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


class CharacterTokenizer(TokenizerABC):

    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        """

        :param text:
        :return:
        """

        return [char for char in text.lower()]


if __name__ == "__main__":
    logging.info('')

    example = "Hello world!"
    tokenizer = CustomTokenizer()
    tokenized = tokenizer.tokenize(text=example)
    logger.info(f"{tokenized}")
