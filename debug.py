import pathlib
from models.Transformer.tokenizer.tokenizer import ExampleTokenizer
sentence = "hello 世界，你好, python"
tokenizer = ExampleTokenizer(vocab_file_path=pathlib.Path("./tokenizer/bert_base_chinese_tokenizer/vocab.txt"))
tokens = tokenizer.tokenize(text=sentence)
print(tokens)