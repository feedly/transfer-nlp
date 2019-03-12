"""
CNN over character-lebvel one-hot encoding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders.vectorizers import NewsVectorizer, Vectorizer
from models.perceptrons import preprocess


class SurnameClassifierCNN(nn.Module):

    def __init__(self, initial_num_channels: int, num_classes: int, num_channels: int):

        super(SurnameClassifierCNN, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) -> torch.Tensor:
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_surname.shape should be (batch, initial_num_channels, max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        features = self.convnet(x_in).squeeze(dim=2)

        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector


class NewsClassifier(nn.Module):

    def __init__(self, embedding_size: int, num_embeddings: int, num_channels: int,
                 hidden_dim: int, num_classes: int, dropout_p: float,
                 pretrained_embeddings: np.array=None, padding_idx: int=0):

        super(NewsClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb: nn.Embedding = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb: nn.Embedding = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )

        self._dropout_p: float = dropout_p
        self.fc1: nn.Linear = nn.Linear(num_channels, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) -> torch.Tensor:
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)

        features = self.convnet(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)

        # mlp classifier
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector


def predict_category(title: str, model: NewsClassifier, vectorizer: Vectorizer, max_length: int):
    """Predict a News category for a new title

    Args:
        title (str): a raw title string
        classifier (NewsClassifier): an instance of the trained classifier
        vectorizer (NewsVectorizer): the corresponding vectorizer
        max_length (int): the max sequence length
            Note: CNNs are sensitive to the input data tensor size.
                  This ensures to keep it the same size as the training data
    """
    title = preprocess(text=title)
    vectorized_title = torch.tensor(vectorizer.vectorize(title=title, vector_length=max_length))
    result = model(x_in=vectorized_title.unsqueeze(0), apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    predicted_category = vectorizer.target_vocab.lookup_index(indices.item())

    return {
        'category': predicted_category,
        'probability': probability_values.item()}



