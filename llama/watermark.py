import torch
from scipy.stats import norm

class AaronsonWatermarker:
    def __init__(self, 
                 hash_key=7, 
                 hashing_schema="prev", 
                 vocab_size = -1, 
                 tokenizer = None):
        self.hash_key = hash_key
        self.hashing_schema = hashing_schema
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def sample(self, probs, tokens, cur_pos):
        # Generate random seed based on the previous token and hash_key
        if self.hashing_schema == "prev": 
            prev_tokens = tokens[:,cur_pos-1]
            # Generate random seeds for each sequence in the batch
            random_seeds = prev_tokens * self.hash_key
        
        # Initialize an array to store the next tokens for each sequence in the batch
        next_tokens = torch.zeros(prev_tokens.shape, dtype=torch.long)
        
        for i, seed in enumerate(random_seeds):
            # Generate uniform random sample r for the i-th sequence
            torch.manual_seed(seed.item())
            r = torch.rand_like(probs[i],dtype = torch.float32)

            # Perform the Aaronson sampling for the i-th sequence
            next_tokens[i] = torch.argmax(torch.log(r) / probs[i])
        return next_tokens

    def detect(self, answer_str, eps = 1e-8):
        tokens = self.tokenizer.encode(answer_str, bos = False, eos = True)
        #tokens: List[int]
        if self.hashing_schema == "prev": 
            prev_tokens = tokens[:-1]  # Exclude the last token
            random_seeds = [prev_tokens[i]*self.hash_key for i in range(len(prev_tokens))]

        T = len(tokens) - 1  # Exclude the first token as per the requirement
        S_T = 0.0

        rti_list = []
        for t, seed in enumerate(random_seeds):
            if t == T:
                break
            torch.manual_seed(seed)
            r = torch.rand(self.vocab_size, dtype = torch.float32)
            r_ti = r[tokens[t+1]] - eps  # Assuming a single token, so directly using r
            rti_list.append(r_ti.item())
            S_T += torch.log(1 / (1 - r_ti))

        # print ("rti:", rti_list)
        # Calculate the Z statistic
        Z = ((S_T / T) - 1) * (T ** 0.5)

        # Calculate the p-value
        p_value = (1 - norm.cdf(Z.item()))

        return p_value, S_T/T, Z

