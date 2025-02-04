# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    watermark: str = "aaronson",
    hashing_schema: str = "lefthash",
    temperature: float = 0.6, #default was 0.6
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        watermark = watermark, 
        hashing_schema = hashing_schema,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=False,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        #print (f"> {result['logprobs']}")
        print("\n==================================\n")

    #Detection
    if watermark: 
        for result in results:
            test = generator.watermarker.detect(result['generation'])
            print ("p-value:",test[0], "ST (null is 1):", test[1], "Z-score", test[2])

if __name__ == "__main__":
    fire.Fire(main)
