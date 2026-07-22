import torch

class ExampleTokenizer:
    def __init__(self,vocab_file_path,do_lower_case=True,strip_accents=None,max_input_chars_per_word=100):
        """
        :param vocab_file_path: vocab.txt file path
        :param do_lower_case: 是否大写转化为小写
        :param strip_accents: 是否去除重音
        :param max_input_chars_per_word: 单个 Basic Token 允许进行 WordPiece 拆分的最大字符数，比如playing，七个字符，可以继续拆分
        """
        self.vocab_file_path = vocab_file_path
        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.max_input_chars_per_word = max_input_chars_per_word

        SPECIAL_TOKENS = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]

        if not self.vocab_file_path.is_file():
            raise FileNotFoundError(f'没有找到文件喵！ {self.vocab_file_path}')
        if not isinstance(self.do_lower_case, bool):
            raise ValueError(f"self.do_lower_case应该为bool值喵！现在它是{self.do_lower_case}")
        if type(self.strip_accents) not in (None, bool):
            raise ValueError(f"self.strip_accents应该为bool值或None值喵！现在它是{self.strip_accents}")
        if not type(self.max_input_chars_per_word) is int:
            raise ValueError(f"self.max_input_chars_per_word应该为int值喵！，现在它是{self.max_input_chars_per_word}")

        self.token_to_id = {}
        self.id_to_token = []

        with open(self.vocab_file_path, "r", encoding="utf-8") as f:
            for token_id, line in enumerate(f):
                token = line.rstrip("\r\n")
                if token == '':
                    raise ValueError(f"{self.vocab_file_path} 该路径下的索引为 {token_id} 的token是空的喵~")
                if token in self.token_to_id:
                    raise ValueError(f"索引为 {token_id} 的 {token} 这一token出现重复了喵，请检查 {self.vocab_file_path} 的文件是否有问题喵~")
                self.token_to_id[token] = token_id
                self.id_to_token.append(token)

        self.vocab_size = len(self.id_to_token)

        if not all(token in self.id_to_token for token in SPECIAL_TOKENS):
            raise ValueError(f"没有在文件中找到特殊token:{SPECIAL_TOKENS},请仔细检查文件喵！")