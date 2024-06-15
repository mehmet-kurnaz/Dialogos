from unittest import result
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import pipeline

class GPT2Assistant:
    def __init__(self):
        # Initialize the model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1

        self.model = GPT2LMHeadModel.from_pretrained("PythonApplication1/PythonApplication1/fine_tuned2_model").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("PythonApplication1/PythonApplication1/fine_tuned2_model")
        self.text_generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=self.device)
        # Default generation parameters
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.9
        self.num_beams = 5

    def set_generation_parameters(self, temperature=1.0, top_k=50, top_p=0.9, num_beams=5):
        # Set the generation parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

    def generate_response(self, instruction):
        # Prepare the instruction prompt
        instruction_prompt = f"### Kullanici:\n{instruction}\n### Asistan:\n"

        # Configure the generation parameters
        generator_config = {
            "max_new_tokens": 124,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "no_repeat_ngram_size":2
        }

        # Generate text based on the instruction prompt and parameters
        result = self.text_generator(instruction_prompt, **generator_config)
        
        generated_response = result[0]['generated_text']
        return generated_response
            
