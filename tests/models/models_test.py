from typing import Dict, Any
import unittest
import torch
import logging

from transfer_nlp.models.perceptrons import MultiLayerPerceptron
from transfer_nlp.models.cnn import SurnameClassifierCNN, NewsClassifier

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(exit=False)