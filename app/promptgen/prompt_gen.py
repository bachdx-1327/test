import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os 

from promptgen.base import PromptGeneration

class MagicGeneration(PromptGeneration):
    """
    Using magic Gen to get prompt generater
    Link: https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion
    """

    def __init__(self, device): 
        """
        Args:
            device (torch.device): Device used.
        """
        super().__init__(device)

    def load_checkpoint(self, checkpoint_name="Gustavosta/MagicPrompt-Stable-Diffusion"):
        model_path = "model/"
        model_chk = os.path.exists('model/pytorch_model.bin')

        if model_chk is False:
            dir_chk = os.path.exists('model/')
            
            if dir_chk is False:
                os.makedirs('model')
            
            repo = checkpoint_name
            tokenizer_dl = GPT2Tokenizer.from_pretrained(repo)
            model_dl = GPT2LMHeadModel.from_pretrained(repo)
            tokenizer_dl.save_pretrained(model_path)
            model_dl.save_pretrained(model_path)
            print('model cloned from Hugging Face')

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=self.tokenizer.eos_token_id, torch_dtype=torch.float32)

    def generate_prompt(self, prompt):
        prompt = prompt    # the beginning of the prompt
        temperature = 0.9             # a higher temperature will produce more diverse results, but with a higher risk of less coherent text
        top_k = 8                     # the number of tokens to sample from at each step
        max_length = 80               # the maximum number of tokens for the output of the model
        repitition_penalty = 1.2      # the penalty value for each repetition of a token
        num_return_sequences = 4       # the number of results to generate

        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        output = self.model.generate(input_ids, do_sample=True, temperature=temperature, top_k=top_k, max_length=max_length, num_return_sequences=num_return_sequences, repetition_penalty=repitition_penalty, penalty_alpha=0.6, no_repeat_ngram_size=1, early_stopping=True)

        lst_prompt = []
        for i in range(len(output)):
            decode_output = self.tokenizer.decode(output[i], skip_special_tokens=True)
            lst_prompt.append(decode_output)
        
        return lst_prompt
