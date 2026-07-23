import torch
import unicodedata

class ExampleTokenizer:
    def __init__(self,vocab_file_path,do_lower_case=True,strip_accents=None,max_input_chars_per_word=100):
        """
        初始化，把文字变成token并储存token_to_id与id_to_token
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
        if not isinstance(self.strip_accents, (bool, type(None))):
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

        # 特殊token及其id
        self.pad_token = "[PAD]"
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token = "[UNK]"
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.cls_token = "[CLS]"
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.sep_token = "[SEP]"
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.mask_token = "[MASK]"
        self.mask_token_id = self.token_to_id[self.mask_token]

    def __len__(self):
        return self.vocab_size

    def _clean_text(self, text):
        """
        替换或删除文本中的特殊字符
        :param text: 原始文本，字符串
        :return: clean_text，清洗后的字符串
        """
        UNUSED_CHARS = {"\x00","\ufffd"}
        EXCEPTION_CONTROL_CHARS = {"\n","\t","\r"}
        if not isinstance(text, str):
            raise ValueError(f"text的数据类型必须要是字符串类型喵")
        clean_text = []
        for char in text:
            unicode_type = unicodedata.category(char)
            if char in UNUSED_CHARS:
                pass
            elif char in EXCEPTION_CONTROL_CHARS:
                clean_text.append(" ")
            elif unicode_type == "Cc" or unicode_type == "Cf":
                pass
            elif unicode_type == "Zs":
                clean_text.append(" ")
            else:
                clean_text.append(char)
        clean_text = "".join(clean_text)
        return clean_text

    def _is_chinese_char(self, char):
        """
        判断单个字符是否在中文区间内
        0x4E00～0x9FFF：CJK Unified Ideographs；
        0x3400～0x4DBF：CJK Extension A；
        0x20000～0x2A6DF：CJK Extension B；
        0x2A700～0x2B73F：CJK Extension C；
        0x2B740～0x2B81F：CJK Extension D；
        0x2B820～0x2CEAF：CJK Extension E/F；
        0xF900～0xFAFF：CJK Compatibility Ideographs；
        0x2F800～0x2FA1F：CJK Compatibility Supplement。
        :param char: 长度为1的字符串，单个字符
        :return: is_chinese, bool，判断是否为中文
        """
        if not len(char) == 1:
            raise ValueError(f"必须要单个字符喵~即长度为1的字符串喵~")
        CHINESE_RANGES = (
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF),
            (0x2F800, 0x2FA1F),
        )
        codepoint = ord(char)
        is_chinese = any(start <= codepoint <= end for start,end in CHINESE_RANGES)
        return is_chinese

    def _add_spaces_around_chinese_chars(self, text):
        """
        将用户输入的文本里面的中文两侧加上空格
        :param text: 用户外部输入的文本
        :return: tokenized_text 加上空格后的中文文本
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的文本必须要是字符串类型的喵！")
        processed_text = []
        for char in text:
            if self._is_chinese_char(char):
                processed_text.extend([" ",char," "])
            else:
                processed_text.append(char)
        processed_text = "".join(processed_text)
        return processed_text

    def _whitespace_tokenize(self, text):
        """
        空白切分方法，把含有空白字符串转换为列表
        :param text: 原始字符串
        :return: token_list，列表，进行空白切分后的列表
        """
        if not isinstance(text, str):
            raise ValueError(f"text的数据类型必须为字符串类型喵！")
        token_list = text.split()
        return token_list

    def _strip_accents(self, token):
        """
        去重音方法，去除单个basic token里面的重音
        :param token: 原始字符串
        :return: clean_token 处理后的字符串
        """
        if not isinstance(token, str):
            raise ValueError(f"token的数据类型必须为字符串类型喵！")
        token = unicodedata.normalize('NFD', token)
        clean_token = []
        for char in token:
            unicode_type = unicodedata.category(char)
            if unicode_type == "Mn":
                pass
            else:
                clean_token.append(char)
        clean_token = "".join(clean_token)
        return clean_token

    def _is_punctuation(self, char):
        """
        标点符号判断方法
        ASCII 标点范围直接判断：
        33–47
        58–64
        91–96
        123–126
        :param char: 输入的原始单个字符
        :return: is_punctuation: bool
        """
        if not len(char) == 1:
            raise ValueError(f"必须要单个字符喵~即长度为1的字符串喵~")
        codepoint = ord(char)
        PUNCTUATION_RANGES = (
            (33,47),
            (58,64),
            (91,96),
            (123,126),
        )
        is_punctuation = any(
            start <= codepoint <= end or unicodedata.category(char).startswith("P")
            for start,end in PUNCTUATION_RANGES
        )
        return is_punctuation

    def _split_on_punctuation(self, token):
        """
        按字符切分字符串为token的列表
        :param token: 输入的原始token
        :return: final_tokens: 处理标点符号后的token列表
        """
        if not isinstance(token, str):
            raise ValueError(f"输入的token必须要是字符串类型的喵！")
        tmp_tokens = []
        final_tokens = []
        for char in token:
            if self._is_punctuation(char):
                if len(tmp_tokens) > 0:
                    final_tokens.append("".join(tmp_tokens))
                final_tokens.append(char)
                tmp_tokens = []
            else:
                tmp_tokens.append(char)
        if len(tmp_tokens) > 0:
            final_tokens.append("".join(tmp_tokens))
        return final_tokens

    def _basic_tokenize(self, text):
        """
        完成基础的切分，返回token列表
        :param text: 用户输入的原始字符串
        :return: token_list
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的内容必须是字符串类型的喵！")
        # 字符串清洗
        clean_text = self._clean_text(text)
        # 在汉字两侧添加空格
        clean_text = self._add_spaces_around_chinese_chars(clean_text)
        # 处理空格
        token_list = self._whitespace_tokenize(clean_text)
        final_token_list = []
        for token in token_list:
            if self.do_lower_case:
                token = token.lower()
            if self.strip_accents or (self.strip_accents is None and self.do_lower_case):
                token = self._strip_accents(token)
            tokens = self._split_on_punctuation(token)
            final_token_list.extend(tokens)
        return final_token_list

    def _wordpiece_tokenize(self, token):
        """
        对单个token进行wordpiece切分
        :param token: 单个basic_token
        :return: sub_tokens: list，进行wordpiece切分后的token list
        """
        if not isinstance(token, str):
            raise ValueError(f"传入的token必须为字符串类型喵")
        if token in self.token_to_id:
            return [token]
        else:
            if len(token) > self.max_input_chars_per_word:
                return ["[UNK]"]
            else:
                start = 0   #还未开始拆分时的起始位置
                sub_tokens = [] #已经匹配成功的wordpiece
                is_bad = False  #是否出现无法匹配的部分
                while start < len(token):
                    end = len(token)
                    current_sub_token = ""
                    while start < end:
                        tmp_token = token[start:end]
                        if start > 0:
                            tmp_token = "##" + tmp_token
                        if tmp_token in self.token_to_id:
                            sub_tokens.append(tmp_token)
                            current_sub_token = tmp_token
                            break
                        else:
                            end-=1
                    if not current_sub_token:
                        is_bad = True
                        break
                    else:
                        start = end
                if is_bad:
                    return ["[UNK]"]
                else:
                    return sub_tokens

    def tokenize(self,text):
        """
        tokenize主方法
        :param text: 用户输入的原始文本,str
        :return: final_token_list, list，tokenize的最终结果
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的text必须是字符串类型的喵！")
        basic_token_list = self._basic_tokenize(text)
        final_token_list = []
        for token in basic_token_list:
            wordpiece_token_list = self._wordpiece_tokenize(token)
            final_token_list.extend(wordpiece_token_list)
        return final_token_list

    def convert_tokens_to_ids(self, tokens):
        """
        将token转为id
        :param tokens: list, token列表
        :return: ids_list: list, 每个token对应的id组成的列表
        """
        if not isinstance(tokens, list):
            raise ValueError(f"传入的token列表必须为list数据类型喵~")
        ids_list = []
        for token in tokens:
            if token in self.token_to_id:
                ids_list.append(self.token_to_id[token])
            else:
                ids_list.append(self.token_to_id["[UNK]"])
        return ids_list

    def convert_ids_to_tokens(self, ids_list):
        """
        将id转为token
        :param ids_list: list, id列表
        :return: tokens_list: list, 每个id对应的token组成的列表
        """
        if not isinstance(ids_list, list):
            raise ValueError(f"传入的id列表必须为list数据类型喵~")
        if not all(type(idx) is int for idx in ids_list):
            raise ValueError(f"传入的id列表中的id必须为整型数据类型喵~")
        if not all(0 <= idx < self.vocab_size for idx in ids_list):
            raise ValueError(f"传入的id列表有异常数据喵~请检查是否存在异常数据")
        tokens_list = []
        for idx in ids_list:
            tokens_list.append(self.id_to_token[idx])
        return tokens_list


