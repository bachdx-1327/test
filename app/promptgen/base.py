from abc import abstractmethod


class PromptGeneration:
    """
    Using LLMs to generate prompt for the model Stable Diffusion.
    """

    def __init__(self, device): 
        """
        Args:
            device (str): device to use for the LLM.
        """
        self.device = device

    @abstractmethod
    def load_checkpoint(self): 
        raise NotImplementedError("Please implement load checkpoint method!")
    
    @abstractmethod
    def generate_prompt(self, prompt): 
        raise NotImplementedError("Please implement generate prompt method!")
    