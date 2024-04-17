# gpt-2-quickstart

# 安装Transformers库
!pip install transformers

# 导入所需的库
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 将模型设置为评估模式
model.eval()

# 准备输入文本
input_text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# 解码并打印输出文本
print(tokenizer.decode(output[0], skip_special_tokens=True))

