from llama import Llama
from llama.tokenizer import Tokenizer
from llama.watermark import AaronsonWatermarker

tokenizer_path='tokenizer.model'
hashing_schema='lefthash'
tokenizer = Tokenizer(model_path=tokenizer_path)

watermarker = AaronsonWatermarker(
    vocab_size = tokenizer.n_words,
    tokenizer = tokenizer,
    hashing_schema = hashing_schema,
)

import json

# 指定 JSON 文件路径
input_file_path = "chat_output_aaronson.json"

# 打开 JSON 文件并解析数据
with open(input_file_path, "r") as input_file:
    output_list = json.load(input_file)

# 循环遍历输出列表中的每个字典
for item in output_list[:]:
    output=item['result']['generation']['content']
    test = watermarker.detect(output)
    print ("p-value:",test[0], "ST (null is 1):", test[1], "Z-score", test[2])