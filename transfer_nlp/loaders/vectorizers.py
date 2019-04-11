import string
from collections import Counter
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

from transfer_nlp.common.tokenizers import CustomTokenizer, CharacterTokenizer
from transfer_nlp.loaders.vocabulary import Vocabulary, CBOWVocabulary, SequenceVocabulary


class Vectorizer:

    def __init__(self, data_vocab: Vocabulary, target_vocab: Vocabulary):

        self.data_vocab: Vocabulary = data_vocab
        self.target_vocab: Vocabulary = target_vocab

    @classmethod
    def from_serializable(cls, contents) -> 'Vectorizer':

        data_vocab = Vocabulary.from_serializable(contents=contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents=contents['target_vocab'])

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    def to_serializable(self) -> Dict[str, Any]:

        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}

    def vectorize(self, input_string: str) -> np.array:

        raise NotImplementedError


class ReviewsVectorizer(Vectorizer):

    def __init__(self, data_vocab: Vocabulary, target_vocab: Vocabulary):

        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CustomTokenizer()

    def vectorize(self, input_string: str) -> np.array:

        encoding = np.zeros(len(self.data_vocab), dtype=np.float32)
        tokens = self.tokenizer.tokenize(text=input_string)

        for token in tokens:
            index = self.data_vocab.lookup_token(token=token)
            encoding[index] = 1

        return encoding

    @classmethod
    def from_dataframe(cls, review_df: pd.DataFrame, cutoff: int = 25) -> Vectorizer:
        tokenizer = CustomTokenizer()

        data_vocab = Vocabulary(add_unk=True)
        target_vocab = Vocabulary(add_unk=False)

        # Add targets to vocabulary
        for rating in sorted(set(review_df.rating)):
            target_vocab.add_token(token=rating)

        # Add tokens from data to vocab
        word_counts = Counter()
        for review in review_df.review:
            for token in tokenizer.tokenize(text=review):
                if token not in string.punctuation:
                    word_counts[token] += 1

        for word in word_counts:
            if word_counts[word] > cutoff:
                data_vocab.add_token(token=word)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)


class SurnamesVectorizer(Vectorizer):

    def __init__(self, data_vocab: Vocabulary, target_vocab: Vocabulary):

        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CharacterTokenizer()

    def vectorize(self, input_string: str) -> np.array:

        encoding = np.zeros(shape=len(self.data_vocab), dtype=np.float32)
        tokens = self.tokenizer.tokenize(text=input_string)
        for character in tokens:
            encoding[self.data_vocab.lookup_token(token=character)] = 1

        return encoding

    @classmethod
    def from_dataframe(cls, surnames_df: pd.DataFrame) -> Vectorizer:

        tokenizer = CharacterTokenizer()

        data_vocab = Vocabulary(unk_token='@')
        target_vocab = Vocabulary(add_unk=False)

        # Add surnames and nationalities to vocabulary
        for index, row in surnames_df.iterrows():

            surname = row.surname
            nationality = row.nationality
            data_vocab.add_many(tokens=tokenizer.tokenize(text=surname))
            target_vocab.add_token(token=nationality)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)


class SurnamesVectorizerCNN(Vectorizer):

    def __init__(self, data_vocab: Vocabulary, target_vocab: Vocabulary, max_surname: int):

        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self._max_surname = max_surname
        self.tokenizer = CharacterTokenizer()

    def vectorize(self, input_string: str) -> np.array:

        encoding = np.zeros(shape=(len(self.data_vocab), self._max_surname), dtype=np.float32)
        tokens = self.tokenizer.tokenize(text=input_string)
        for char_index, character in enumerate(tokens):
            encoding[self.data_vocab.lookup_token(token=character)][char_index] = 1

        return encoding

    @classmethod
    def from_dataframe(cls, surnames_df: pd.DataFrame) -> Vectorizer:

        tokenizer = CharacterTokenizer()

        data_vocab = Vocabulary(unk_token='@')
        target_vocab = Vocabulary(add_unk=False)
        max_surname = 0

        # Add surnames and nationalities to vocabulary
        for index, row in surnames_df.iterrows():

            surname = row.surname
            nationality = row.nationality
            data_vocab.add_many(tokens=tokenizer.tokenize(text=surname))
            target_vocab.add_token(token=nationality)
            max_surname = max(max_surname, len(surname))

        return cls(data_vocab=data_vocab, target_vocab=target_vocab, max_surname=max_surname)

    @classmethod
    def from_serializable(cls, contents):

        data_vocab = Vocabulary.from_serializable(contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])
        return cls(data_vocab=data_vocab, target_vocab=target_vocab,
                   max_surname=contents['max_surname_length'])

    def to_serializable(self):

        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable(),
                'max_surname_length': self._max_surname}


class CBOWVectorizer(Vectorizer):

    def __init__(self, data_vocab: CBOWVocabulary, target_vocab: CBOWVocabulary):

        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.data_vocab: CBOWVocabulary = data_vocab
        self.target_vocab: CBOWVocabulary = target_vocab
        self.tokenizer = CustomTokenizer()

    def vectorize(self, context: str, vector_length: int=-1) -> np.array:

        tokens = self.tokenizer.tokenize(text=context)
        indices = [self.data_vocab.lookup_token(token) for token in tokens]
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, cbow_df: pd.DataFrame) -> Vectorizer:

        tokenizer = CustomTokenizer()

        data_vocab = CBOWVocabulary()
        for index, row in cbow_df.iterrows():
            for token in tokenizer.tokenize(text=row.context):
                data_vocab.add_token(token)
                data_vocab.add_token(row.target)

        return cls(data_vocab=data_vocab, target_vocab=CBOWVocabulary())

    @classmethod
    def from_serializable(cls, contents) -> Vectorizer:

        data_vocab = Vocabulary.from_serializable(contents['data_vocab'])
        return cls(data_vocab=data_vocab, target_vocab=CBOWVocabulary())

    def to_serializable(self) -> Dict:

        return {'data_vocab': self.data_vocab.to_serializable()}


class NewsVectorizer(Vectorizer):

    def __init__(self, data_vocab: SequenceVocabulary, target_vocab: Vocabulary):

        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CustomTokenizer()

    def vectorize(self, title: str, vector_length: int=-1) -> np.array:

        tokens = self.tokenizer.tokenize(text=title)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, news_df: pd.DataFrame, cutoff=25) -> Vectorizer:

        tokenizer = CustomTokenizer()

        target_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(news_df.category)):
            target_vocab.add_token(category)

        word_counts = Counter()
        for title in news_df.title:
            for token in tokenizer.tokenize(text=title):
                if token not in string.punctuation:
                    word_counts[token] += 1

        data_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                data_vocab.add_token(word)

        return cls(data_vocab, target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        data_vocab = SequenceVocabulary.from_serializable(contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    def to_serializable(self):

        return {
            'data_vocab': self.data_vocab.to_serializable(),
            'target_vocab': self.target_vocab.to_serializable()}


class SurnameVectorizerRNN(Vectorizer):

    def __init__(self, data_vocab: SequenceVocabulary, target_vocab: Vocabulary):
        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CharacterTokenizer()

    def vectorize(self, surname: str, vector_length: int=-1) -> Tuple[np.array, int]:

        tokens = self.tokenizer.tokenize(text=surname)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, surname_df: pd.DataFrame) -> Vectorizer:
        tokenizer = CharacterTokenizer()

        data_vocab = SequenceVocabulary()
        target_vocab = Vocabulary(add_unk=False)

        for index, row in surname_df.iterrows():
            data_vocab.add_many(tokens=tokenizer.tokenize(text=row.surname))
            target_vocab.add_token(row.nationality)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        data_vocab = SequenceVocabulary.from_serializable(contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    def to_serializable(self):
        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}


class SurnameVectorizerGeneration(Vectorizer):

    def __init__(self, data_vocab: SequenceVocabulary, target_vocab: Vocabulary):
        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CharacterTokenizer()

    def vectorize(self, surname: str, vector_length: int=-1) -> Tuple[np.array, np.array]:

        tokens = self.tokenizer.tokenize(text=surname)

        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        from_vector = np.empty(shape=vector_length, dtype=np.int64)
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self.data_vocab.mask_index

        to_vector = np.empty(shape=vector_length, dtype=np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.data_vocab.mask_index

        return from_vector, to_vector

    @classmethod
    def from_dataframe(cls, surname_df: pd.DataFrame) -> Vectorizer:

        tokenizer = CharacterTokenizer()

        data_vocab = SequenceVocabulary()
        target_vocab = Vocabulary(add_unk=False)

        for index, row in surname_df.iterrows():
            data_vocab.add_many(tokens=tokenizer.tokenize(row.surname))
            target_vocab.add_token(row.nationality)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        data_vocab = SequenceVocabulary.from_serializable(contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    def to_serializable(self):
        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}


class FeedlyVectorizer(Vectorizer):

    def __init__(self, data_vocab: SequenceVocabulary, target_vocab: Vocabulary):
        super().__init__(data_vocab=data_vocab, target_vocab=target_vocab)
        self.tokenizer = CustomTokenizer()

    def vectorize(self, content: str, vector_length: int=-1) -> Tuple[np.array, np.array]:

        content = self.tokenizer.tokenize(text=content)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in content)
        indices.append(self.data_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        from_vector = np.empty(shape=vector_length, dtype=np.int64)
        from_indices = indices[:-1]
        from_vector[:len(from_indices)] = from_indices
        from_vector[len(from_indices):] = self.data_vocab.mask_index

        to_vector = np.empty(shape=vector_length, dtype=np.int64)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.data_vocab.mask_index

        return from_vector, to_vector

    @classmethod
    def from_dataframe(cls, feedly_df: pd.DataFrame, cutoff: int = 10) -> Vectorizer:

        data_vocab = SequenceVocabulary()
        target_vocab = Vocabulary(add_unk=False)

        # Add tokens to reviews vocab
        word_counts = Counter()
        for article in feedly_df.content:
            for token in tokenize(text=article):
                if token not in string.punctuation:
                    word_counts[token] += 1

        for word in word_counts:
            if word_counts[word] > cutoff:
                data_vocab.add_token(token=word)

        for index, row in feedly_df.iterrows():
            target_vocab.add_token(row.nationality)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    @classmethod
    def from_serializable(cls, contents):
        data_vocab = SequenceVocabulary.from_serializable(contents['data_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(data_vocab=data_vocab, target_vocab=target_vocab)

    def to_serializable(self):
        return {'data_vocab': self.data_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}


class NMTVectorizer(object):

    def __init__(self, data_vocab: SequenceVocabulary, target_vocab: SequenceVocabulary, max_source_length: int, max_target_length: int):

        self.data_vocab: SequenceVocabulary = data_vocab
        self.target_vocab: SequenceVocabulary = target_vocab

        self.max_source_length: int = max_source_length
        self.max_target_length: int = max_target_length

        self.source_tokenizer = CustomTokenizer()
        self.target_tokenizer = CustomTokenizer()

    def _vectorize(self, indices: List[int], vector_length: int=-1, mask_index: int=0) -> np.array:

        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(shape=vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text: str) -> List[int]:

        tokens = self.source_tokenizer.tokenize(text=text)

        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token) for token in tokens)
        indices.append(self.data_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text: str) -> Tuple[List[int], List[int]]:

        tokens = self.target_tokenizer.tokenize(text=text)

        indices = [self.target_vocab.lookup_token(token) for token in tokens]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices

    def vectorize(self, source_text: str, target_text: str, use_dataset_max_lengths: bool=True) -> Dict[str, Any]:

        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.data_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)
        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices)}

    @classmethod
    def from_dataframe(cls, bitext_df: pd.DataFrame) -> 'NMTVectorizer':

        data_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        source_tokenizer = CustomTokenizer()
        target_tokenizer = CustomTokenizer()

        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = source_tokenizer.tokenize(row["source_language"])
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            data_vocab.add_many(source_tokens)

            target_tokens = target_tokenizer.tokenize(row["target_language"])
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            target_vocab.add_many(target_tokens)

        return cls(data_vocab=data_vocab, target_vocab=target_vocab, max_source_length=max_source_length, max_target_length=max_target_length)

    @classmethod
    def from_serializable(cls, contents) -> 'NMTVectorizer':
        data_vocab = SequenceVocabulary.from_serializable(contents["data_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(data_vocab=data_vocab,
                   target_vocab=target_vocab,
                   max_source_length=contents["max_source_length"],
                   max_target_length=contents["max_target_length"])

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "data_vocab": self.data_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length}