from abc import ABC, abstractmethod


class TrainerABC(ABC):

    @abstractmethod
    def train(self):
        pass
