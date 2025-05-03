from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all models
    """

    def __init__(self):
        pass

    @abstractmethod
    def train(self, save_to_hf: bool = False):
        """
        Train the model
        """
        pass

    @abstractmethod
    def forward(self, image_path: str, question: str):
        """
        Forward pass on the model to get an answer
        """
        pass
