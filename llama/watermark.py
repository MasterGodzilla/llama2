import torch
import numpy as np
from scipy.stats import norm

class AaronsonWatermarker:
    def __init__(self, 
                 hash_key = 2971215073, 
                 mod_key = 15485863,
                 hashing_schema = "lefthash", 
                 vocab_size = -1, 
                 tokenizer = None,):
        self.hash_key = hash_key
        self.mod_key = mod_key
        self.hashing_schema = hashing_schema
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def sample(self, probs, tokens, cur_pos, k=4):
        # Generate random seed based on the previous token and hash_key
        if self.hashing_schema == "lefthash": 
            prev_tokens = tokens[:,cur_pos-1]
            # Generate random seeds for each sequence in the batch
            random_seeds = prev_tokens * self.hash_key % self.mod_key 
        elif self.hashing_schema == "minhash":
            prev_tokens = tokens[:, cur_pos - k:cur_pos]
            #print ("prev_tokens",prev_tokens)
            hashed_tokens = prev_tokens * self.hash_key % self.mod_key
            #print ("hashed_tokens",hashed_tokens)
            random_seeds, _ = torch.min(hashed_tokens,dim = 1)
            #print ("t:",cur_pos, "random_seeds",random_seeds[0])
        
        # Initialize an array to store the next tokens for each sequence in the batch
        next_tokens = torch.zeros(tokens.shape[0], dtype=torch.long)
        
        for i, seed in enumerate(random_seeds):
            # Generate uniform random sample r for the i-th sequence
            #torch.manual_seed(seed.item())
            #print ("sample_seed",seed)
            #r = torch.rand_like(probs[i],dtype = torch.float32)
            #r = torch.rand(self.vocab_size, dtype = torch.float32)

            #try replace with numpy
            np.random.seed(int(seed.item()))
            # print("sample_seed", seed)
            r_numpy = np.random.rand(self.vocab_size).astype(np.float32)
            r = torch.from_numpy(r_numpy).cuda()
            
            # Perform the Aaronson sampling for the i-th sequence
            next_tokens[i] = torch.argmax(torch.log(r) / probs[i])

            # print ("token", next_tokens[i], "rti:", r[next_tokens[i]],"rt0", r[0])

        return next_tokens

    def detect(self, answer_str, eps = 1e-8, k=4):
        tokens = self.tokenizer.encode(answer_str, bos = False, eos = True)
        #tokens: List[int]
        if self.hashing_schema == "lefthash": 
            prev_tokens = tokens[:-1]  # Exclude the last token
            random_seeds = [prev_tokens[i]*self.hash_key% self.mod_key  for i in range(len(prev_tokens))]
        elif self.hashing_schema == "minhash":
            prev_tokens = tokens[:-1]
            hashed_tokens = [prev_tokens[i]*self.hash_key% self.mod_key  for i in range(len(prev_tokens))]
            random_seeds = [min(hashed_tokens[i:i+k]) for i in range(len(prev_tokens)-k+1)]
            #print ("prev_tokens",prev_tokens)
            #print ("hashed_tokens",hashed_tokens)
            #print ("random_seeds",random_seeds)
            
        T = len(random_seeds)  # Exclude the first token as per the requirement
        S_T = 0.0

        rti_list = []
        for t, seed in enumerate(random_seeds):
            if t == T:
                break
            #torch.manual_seed(seed)
            #print ("detect seed",seed)
            #r = torch.rand(self.vocab_size, dtype = torch.float32) #too low precision cause inf
            
            #try replacing with numpy.rand
            np.random.seed(int(seed))
            #print("detect seed", seed)
            r_numpy = np.random.rand(self.vocab_size).astype(np.float32)
            r = torch.from_numpy(r_numpy).cuda()
            
            if self.hashing_schema == "lefthash": 
                r_ti = r[tokens[t+1]] - eps
                # print ("token", tokens[t+1],"rti", r_ti, "rt0", r[0])
            elif self.hashing_schema == "minhash": 
                r_ti = r[tokens[t+k]] - eps
                # print ("t+k:", t+k, "seed:", seed)
            rti_list.append(r_ti.item())
            S_T += torch.log(1 / (1 - r_ti))

        # print ("rti:", rti_list)
        # Calculate the Z statistic
        Z = ((S_T / T) - 1) * (T ** 0.5)

        # Calculate the p-value
        p_value = (1 - norm.cdf(Z.item()))

        return p_value, S_T.item()/T, Z.item()

