import torch
import numpy as np

class AaronsonWatermarker:
    def __init__(self, hash_key=7, hashing_schema="prev"):
        self.hash_key = hash_key
        self.hashing_schema = hashing_schema

    def sample(self, probs, tokens):
        # Generate random seed based on the previous token and hash_key
        if self.hashing_schema == "prev": 
            prev_tokens = tokens[:,-1]
            # Generate random seeds for each sequence in the batch
            random_seeds = prev_tokens * self.hash_key
        
        # Initialize an array to store the next tokens for each sequence in the batch
        next_tokens = torch.zeros(tokens.shape[0], dtype=torch.long)
        
        for i, seed in enumerate(random_seeds):
            # Generate uniform random sample r for the i-th sequence
            torch.manual_seed(seed.item())
            r = torch.rand_like(probs[i])
            
            # Perform the Aaronson sampling for the i-th sequence
            next_tokens[i] = torch.argmax(torch.log(r) / probs[i])
        
        return next_tokens

    def detection(self, tokens):
        pass  # To be implemented
