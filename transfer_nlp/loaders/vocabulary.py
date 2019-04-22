from typing import Dict


class Vocabulary:

    def __init__(self, token2id: Dict = None, add_unk: bool = True, unk_token: str = "<UNK>"):

        if token2id is None:
            token2id = {}

        self._token2id: Dict = token2id
        self._id2token = {idx: token for token, idx in self._token2id.items()}

        self._add_unk: bool = add_unk
        self._unk_token: str = unk_token
        self.unk_index: int = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    @classmethod
    def from_serializable(cls, contents):

        return cls(**contents)

    def to_serializable(self):

        return {
            'token2id': self._token2id,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token}

    def add_token(self, token: str):

        if token in self._token2id:
            index = self._token2id[token]
        else:
            index = len(self._token2id)
            self._token2id[token] = index
            self._id2token[index] = token
        return index

    def add_many(self, tokens):

        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str):

        if self._add_unk:
            return self._token2id.get(token, self.unk_index)
        else:
            return self._token2id.get(token, None)

    def lookup_index(self, index: int):

        if index not in self._id2token:
            raise ValueError(f"Index {index} is not present in the Vocabulary")

        else:
            return self._id2token[index]

    def __str__(self):
        return f"Vocabulary(size={len(self)})"

    def __len__(self):
        return len(self._token2id)


class CBOWVocabulary(Vocabulary):

    def __init__(self, token2id: Dict = None, add_unk: bool = True, unk_token: str = "<UNK>", mask_token: str = "<MASK>"):
        super().__init__(token2id=token2id, add_unk=add_unk, unk_token=unk_token)
        self._mask_token = mask_token
        self.mask_index = self.add_token(self._mask_token)

    def to_serializable(self):
        contents = super(CBOWVocabulary, self).to_serializable()
        contents.update({
            'mask_token': self._mask_token})
        return contents


class SequenceVocabulary(Vocabulary):

    def __init__(self, token2id: Dict = None, unk_token: str = "<UNK>",
                 mask_token: str = "<MASK>", begin_seq_token: str = "<BEGIN>",
                 end_seq_token: str = "<END>"):

        super(SequenceVocabulary, self).__init__(token2id=token2id, add_unk=True, unk_token=unk_token)

        self._mask_token: str = mask_token
        self._begin_seq_token: str = begin_seq_token
        self._end_seq_token: str = end_seq_token

        self.mask_index: int = self.add_token(self._mask_token)
        self.begin_seq_index: int = self.add_token(self._begin_seq_token)
        self.end_seq_index: int = self.add_token(self._end_seq_token)

    def to_serializable(self):

        contents = super(SequenceVocabulary, self).to_serializable()

        contents.update({
            'mask_token': self._mask_token,
            'begin_seq_token': self._begin_seq_token,
            'end_seq_token': self._end_seq_token})
        del contents['add_unk']
        return contents

    @classmethod
    def from_serializable(cls, contents):

        return cls(**contents)

    def lookup_token(self, token):

        if self.unk_index >= 0:
            return self._token2id.get(token, self.unk_index)
        else:
            return self._token2id[token]
