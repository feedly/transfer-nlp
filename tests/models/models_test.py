from typing import Dict, Any
import unittest
import torch
import logging

from transfer_nlp.models.perceptrons import MultiLayerPerceptron
from transfer_nlp.models.cnn import SurnameClassifierCNN, NewsClassifier
from transfer_nlp.models.rnn import ElmanRNN, SurnameClassifierRNN
from transfer_nlp.models.cbow import CBOWClassifier
from transfer_nlp.models.generation import SurnameConditionedGenerationModel

class MLPTest(unittest.TestCase):

    def test_mlp(self):
        input_dim = 100
        hidden_dim = 10
        output_dim = 5
        batch_size = 20

        model = MultiLayerPerceptron(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        tensor = torch.randn(size=(batch_size, input_dim))
        output = model(x_in=tensor)

        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=output_dim)

    def test_cnn(self):

        batch_size = 20
        initial_num_channels = 10
        max_surname_length = 20
        num_classes = 5
        num_channels = 15

        model = SurnameClassifierCNN(initial_num_channels=initial_num_channels, num_classes=num_classes, num_channels=num_channels)

        tensor = torch.randn(size=(batch_size, initial_num_channels, max_surname_length))
        output = model(x_in=tensor)

        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=num_classes)

        embedding_size = 10
        num_embeddings = 10
        num_channels = 10
        hidden_dim = 10
        num_classes = 10
        dropout_p = 0.5
        max_size = 200

        model = NewsClassifier(embedding_size=embedding_size, num_embeddings=num_embeddings, num_channels=num_channels,
                               hidden_dim=hidden_dim, num_classes=num_classes, dropout_p=dropout_p)

        tensor = torch.randint(low=1, high=num_embeddings, size=(batch_size, max_size))
        output = model(x_in=tensor)

        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=num_classes)

    def test_rnn(self):

        batch_size = 32
        input_size = 100
        hidden_size = 200
        seq_size = 50

        model = ElmanRNN(input_size=input_size, hidden_size=hidden_size)

        tensor = torch.randn(size=(batch_size, seq_size, input_size))
        output = model(x_in=tensor)

        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=tensor.size(dim=1), second=output.size(dim=1))
        self.assertEqual(first=tensor.size(dim=2), second=input_size)
        self.assertEqual(first=output.size(dim=2), second=hidden_size)

        embedding_size = 100
        num_embeddings = 100
        num_classes = 10
        rnn_hidden_size = 64

        model = SurnameClassifierRNN(embedding_size=embedding_size, num_embeddings=num_embeddings, num_classes=num_classes,
                                     rnn_hidden_size=rnn_hidden_size)

        tensor = torch.randint(low=1, high=num_embeddings, size=(batch_size, embedding_size))
        lens = torch.randint(low=1, high=num_embeddings, size=(batch_size,))
        output = model(x_in=tensor, x_lengths=lens)
        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=num_classes)

    def test_cbow(self):

        vocabulary_size = 5000
        embedding_size = 100
        batch_size = 32

        model = CBOWClassifier(vocabulary_size=vocabulary_size, embedding_size=embedding_size)

        tensor = torch.randint(low=1, high=vocabulary_size, size=(batch_size, embedding_size))
        output = model(x_in=tensor, apply_softmax=False)
        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=vocabulary_size)

    def test_generation(self):

        char_embedding_size = 32
        char_vocab_size = 256
        rnn_hidden_size = 200
        num_nationalities = 2
        batch_size = 32
        max_sequence = 100

        model = SurnameConditionedGenerationModel(char_embedding_size=char_embedding_size, char_vocab_size=char_vocab_size, rnn_hidden_size=rnn_hidden_size,
                                                  num_nationalities=num_nationalities, conditioned=True)

        tensor = torch.ones(size=(batch_size, max_sequence)).long()
        nationality_index = torch.zeros(size=(batch_size,), dtype=torch.int64)
        output = model(x_in=tensor, nationality_index=nationality_index)

        self.assertEqual(first=tensor.size(dim=0), second=output.size(dim=0))
        self.assertEqual(first=output.size(dim=1), second=max_sequence)
        self.assertEqual(first=output.size(dim=2), second=char_vocab_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(exit=False)