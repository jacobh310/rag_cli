from abc import abstractmethod, ABC


class Command(ABC):
    """Base Class for CLI commands"""
    
    @abstractmethod
    def run(self):
        """Base Method for Command Base Class"""
        pass
