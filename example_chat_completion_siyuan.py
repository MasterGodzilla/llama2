# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import json
   

from tqdm import tqdm

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    watermark: str = "aaronson",
    hashing_schema: str = "lefthash",
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
                watermark = watermark, 
                hashing_schema = hashing_schema,
    )
    with open("all_dialogs.json", "r") as f:
        all_dialogs = json.load(f)
    output_list = []
    
    
    dialogs=all_dialogs[5000:20000]

     
    batch_size = max_batch_size  # 批处理大小

    # 将所有对话分成批次
    num_batches = len(dialogs) // batch_size
    if len(dialogs) % batch_size != 0:
        num_batches += 1

    output_list = []  # 用于存储输出的列表

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dialogs))
        batch_dialogs = dialogs[start_idx:end_idx]

        results = generator.chat_completion(
            batch_dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(batch_dialogs, results):
            output_dict = {
                "prompt": dialog,
                "result": result,
                'wm': watermark,
            }
            output_list.append(output_dict)
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

        
        
    output_file_path = f"chat_output_{watermark}_5k-20k.json"
    
    with open(output_file_path, "w") as output_file:
        json.dump(output_list, output_file,)
        
    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")
    
    # dialogs = [
    #     [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    #     [
    #         {"role": "system", "content": "Always answer with Haiku"},
    #         {"role": "user", "content": "I am going to Paris, what should I see?"},
    #     ],
    #     [
    #         {
    #             "role": "system",
    #             "content": "Always answer with emojis",
    #         },
    #         {"role": "user", "content": "How to go from Beijing to NY?"},
    #     ],
    # ]


if __name__ == "__main__":
    fire.Fire(main)
