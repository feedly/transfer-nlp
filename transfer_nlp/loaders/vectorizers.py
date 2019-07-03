class Vectorizer:

    def __init__(self, data_file: str):
        self.data_file = data_file

    def vectorize(self, input_string: str):
        raise NotImplementedError
