import pathlib
import torch
from models.Transformer.tokenizer.tokenizer import ExampleTokenizer, QATokenizer
from models.Transformer.embedding.embedding import BertEmbedding

# 验证tokenizer
sentence_a = "hello 世界，你好, python"
sentence_b = "@全体成员 选修张翔老师《软件定义网络》《Python微服务设计与DevOps实践》以及《进阶式挑战性综合项目1》的同学，请查收自己的学生邮箱std.uestc.edu.cn，根据邮箱提示注册登陆思科网络技术学院netacad.com，开展暑期预学习。\n- 选修《Python微服务设计与DevOps实践》与《进阶式挑战性综合项目1》的同学，主要根据自己实际情况按需学习Python和Linux相关辅助课程\n- 选修《软件定义网络》的同学，除按需学习Python和Linux相关辅助课程外，若无网络基础，需在假期完成《网络简介》课程全部内容的学习，相关在线知识考试和在线仿真实验考试会计入平时成绩（参见课程介绍），同时完成辅助课程《DevNet》模块4”了解和使用 API“的学习与在线知识测评。"
tokenizer = ExampleTokenizer(vocab_file_path=pathlib.Path("./tokenizer/bert_base_chinese_tokenizer/vocab.txt"))
encoded_dict = tokenizer.encode(text=sentence_a,text_pair=sentence_b,padding=True,add_special_tokens=True,truncation=True,max_length=128)
print(encoded_dict)
input_ids = encoded_dict['input_ids']
print(input_ids)
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(tokens)
token_type_ids = encoded_dict['token_type_ids']
print(token_type_ids)
attention_mask = encoded_dict['attention_mask']
print(attention_mask)

QAtokenizer = QATokenizer(vocab_file_path=pathlib.Path("./tokenizer/bert_base_chinese_tokenizer/vocab.txt"))
encoded_dict_with_offset = QAtokenizer.encode_with_offsets(text=sentence_a,text_pair=sentence_b,padding=True,add_special_tokens=True,truncation=True,max_length=128)
print(encoded_dict_with_offset)
input_ids_with_offset = encoded_dict_with_offset['input_ids']
print(input_ids_with_offset)
tokens_with_offset = QAtokenizer.convert_ids_to_tokens(input_ids_with_offset)
print(tokens_with_offset)
token_type_ids_with_offset = encoded_dict_with_offset['token_type_ids']
print(token_type_ids_with_offset)
attention_mask_with_offset = encoded_dict_with_offset['attention_mask']
print(attention_mask_with_offset)
offset_mapping = encoded_dict_with_offset['offset_mapping']
print(offset_mapping)

# 验证embedding
input_ids = torch.tensor(
    [
        [1, 2, 3, 4, 0],
        [5, 6, 7, 0, 0],
    ],
    dtype=torch.long,
)

token_type_ids = torch.tensor(
    [
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=torch.long,
)

vocab_size = 20
hidden_size = 8
max_position_embeddings = 16
type_vocab_size = 2
padding_idx = 0
dropout = 0

embedding = BertEmbedding(vocab_size=vocab_size,hidden_size=hidden_size,max_position_embeddings=max_position_embeddings,type_vocab_size=type_vocab_size,padding_idx=padding_idx,dropout=dropout)
output = embedding(input_ids=input_ids,token_type_ids=token_type_ids)

print(output.shape)
print(torch.isfinite(output).all())

output_without_types = embedding(input_ids)

print(output_without_types.shape)
print(torch.isfinite(output_without_types).all())

loss = output.pow(2).mean()
loss.backward()

print(embedding.word_embeddings.weight.grad is not None)
print(embedding.position_embeddings.weight.grad is not None)
print(embedding.type_embeddings.weight.grad is not None)
print(embedding.layer_norm.weight.grad is not None)
print(embedding.word_embeddings.weight.grad[0])