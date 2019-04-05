import logging
import unittest

from transfer_nlp.loaders.vocabulary import Vocabulary


class MLPTest(unittest.TestCase):

    def test_vocabulary(self):
        voc = Vocabulary()
        self.assertEqual(first=voc.to_serializable(), second={
            'token2id': {
                '<UNK>': 0},
            'add_unk': True,
            'unk_token': '<UNK>'})

        voc = Vocabulary()
        token = 'Feedly'
        voc.add_token(token=token)
        self.assertEqual(first=voc.lookup_token(token=token), second=1)
        self.assertEqual(first=voc.lookup_index(index=1), second=token)

        voc = Vocabulary()
        tokens = ['Feedly', 'NLP']
        voc.add_many(tokens=tokens)
        self.assertEqual(first=voc.lookup_token(token='Feedly'), second=1)
        self.assertEqual(first=voc.lookup_index(index=1), second='Feedly')
        self.assertEqual(first=voc.lookup_token(token='NLP'), second=2)
        self.assertEqual(first=voc.lookup_index(index=2), second='NLP')
        self.assertEqual(first=voc.to_serializable(), second={
            'token2id': {
                '<UNK>': 0,
                'Feedly': 1,
                'NLP': 2},
            'add_unk': True,
            'unk_token': '<UNK>'})

        voc = Vocabulary.from_serializable(contents={
            'token2id': {
                '<UNK>': 0,
                'Feedly': 1,
                'NLP': 2},
            'add_unk': True,
            'unk_token': '<UNK>'})
        self.assertEqual(first=voc.to_serializable(), second={
            'token2id': {
                '<UNK>': 0,
                'Feedly': 1,
                'NLP': 2},
            'add_unk': True,
            'unk_token': '<UNK>'})

        self.assertEqual(first=len(voc), second=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(exit=False)
