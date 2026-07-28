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
        :return: processed_text 加上空格后的中文文本
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
        按字符切分字符串为token的列表，负责处理标点符号
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

    def encode(self, text, add_special_tokens=True, truncation = False, padding=False, text_pair=None, max_length=None):
        """
        tokenizer的编码逻辑
        :param text: str, 源文本
        :param add_special_tokens: bool, 是否加入特殊token
        :param truncation: bool, 超过长度时是否进行截断
        :param padding: bool, 是否进行pad补位
        :param text_pair: None or str, 另一段文本
        :param max_length: int, token最大长度
        :return: encoded_set, dict {
            'input_ids': input_ids, token转化为id后的列表
            'token_type_ids': token_type_ids, token来自于哪个句子
            'attention_mask': attention_mask, 1表示有效Token，0表示Padding}
        """
        # 初始化最终输出
        encoded_dict = {}
        # 参数校验
        if not isinstance(text, str):
            raise ValueError(f"输入的text必须为字符串类型的喵！")
        if not isinstance(add_special_tokens, bool):
            raise ValueError(f"输入的add_special_tokens必须为bool类型的喵！")
        if not isinstance(truncation, bool):
            raise ValueError(f"输入的truncation必须为bool类型的喵！")
        if not isinstance(padding, bool):
            raise ValueError(f"输入的padding必须为bool类型的喵！")
        if not isinstance(text_pair, (str,type(None))):
            raise ValueError(f"输入的text_pair必须为字符串类型或者None类型的喵！")
        if type(max_length) is not int and max_length is not None:
            raise ValueError(f"输入的max_length必须为int类型或者None类型的喵！")
        if type(max_length) is int and max_length <= 0:
            raise ValueError(f"输入的max_length如果为int必须大于0喵！")
        if truncation and max_length is None:
            raise ValueError(f"说了要截断truncation就要告诉我多少才截断max_length喵！")
        if padding and (type(max_length) is not int or max_length <= 0):
            raise ValueError(f"如果要padding填充的话，max_length的值必须至少为正整数喵！")
        # 分词
        token_a = self.tokenize(text)
        if text_pair is not None:
            token_b = self.tokenize(text_pair)
            if add_special_tokens:
                if truncation:
                    special_tokens_count = 3
                    if max_length < special_tokens_count:
                        raise ValueError(f"max_length: {max_length} 的数值太小了喵！")
                    available_length = max_length - special_tokens_count
                    while len(token_a) + len(token_b) > available_length:
                        if len(token_a) > len(token_b):
                            token_a.pop()
                        else:
                            token_b.pop()
                token_list = [self.cls_token] + token_a + [self.sep_token] + token_b + [self.sep_token]
                token_type_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) + 1)
            else:
                if truncation:
                    special_tokens_count = 0
                    available_length = max_length - special_tokens_count
                    while len(token_a) + len(token_b) > available_length:
                        if len(token_a) > len(token_b):
                            token_a.pop()
                        else:
                            token_b.pop()
                token_list = token_a + token_b
                token_type_ids = [0] * len(token_a) + [1] * len(token_b)
        else:
            if add_special_tokens:
                if truncation:
                    special_tokens_count = 2
                    if max_length < special_tokens_count:
                        raise ValueError(f"max_length: {max_length} 的数值太小了喵！")
                    available_length = max_length - special_tokens_count
                    if len(token_a) > available_length:
                        token_a = token_a[:available_length]
                token_list = [self.cls_token] + token_a + [self.sep_token]
                token_type_ids = [0] * (len(token_a) + 2)
            else:
                if truncation:
                    special_tokens_count = 0
                    available_length = max_length - special_tokens_count
                    if len(token_a) > available_length:
                        token_a = token_a[:available_length]
                token_list = token_a
                token_type_ids = [0] * len(token_a)
        # 进一步组装参数
        input_ids = self.convert_tokens_to_ids(token_list)
        attention_mask = [1] * len(input_ids)
        if not (len(input_ids) == len(token_type_ids) == len(attention_mask)):
            raise RuntimeError(f"input_ids, token_type_ids, attention_mask, offset_mapping四个值的长度不一致喵！"
                               f"input_ids长度为：{len(input_ids)}"
                               f"token_type_ids长度为：{len(token_type_ids)}"
                               f"attention_mask长度为：{len(attention_mask)}")
        if padding:
            padding_length = max_length - len(input_ids)
            if padding_length < 0:
                raise ValueError(f"目前input_ids的长度大于max_length，此时如果需要padding请开启truncation喵！")
            input_ids += [self.pad_token_id] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            if not len(token_type_ids) == len(input_ids) == len(attention_mask) == max_length:
                raise ValueError(f"padding时出现异常！请检查源代码！")
        # 加入字典
        encoded_dict["input_ids"] = input_ids
        encoded_dict["token_type_ids"] = token_type_ids
        encoded_dict["attention_mask"] = attention_mask
        return encoded_dict

class QATokenizer(ExampleTokenizer):
    def _clean_text_with_offsets(self, text):
        """
        替换或删除文本中的特殊字符，并生成相应的offset
        :param text: 原始文本，字符串
        :return: clean_text，清洗后的字符串, char_offsets, 给每个字符赋予的offset,list
        """
        UNUSED_CHARS = {"\x00", "\ufffd"}
        EXCEPTION_CONTROL_CHARS = {"\n", "\t", "\r"}
        if not isinstance(text, str):
            raise ValueError(f"text的数据类型必须要是字符串类型喵")
        clean_text = []
        char_offsets = []
        for index,char in enumerate(text):
            unicode_type = unicodedata.category(char)
            if char in UNUSED_CHARS:
                pass
            elif char in EXCEPTION_CONTROL_CHARS:
                clean_text.append(" ")
                char_offsets += [(index, index+1)]
            elif unicode_type == "Cc" or unicode_type == "Cf":
                pass
            elif unicode_type == "Zs":
                clean_text.append(" ")
                char_offsets += [(index, index + 1)]
            else:
                clean_text.append(char)
                char_offsets += [(index, index + 1)]
        clean_text = "".join(clean_text)
        return clean_text,char_offsets

    def _add_spaces_around_chinese_chars_with_offsets(self, text, char_offsets):
        """
        将用户输入的文本里面的中文两侧加上空格，offset在相应的位置处理为(0,0)
        :param text: 用户外部输入的文本
        :param char_offsets: 文本对应的offset序列
        :return: processed_text 加上空格后的中文文本, processed_char_offsets 加上空格后的offset序列
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的文本必须要是字符串类型的喵！")
        if not len(text) == len(char_offsets):
            raise ValueError(f"text的长度和相应的offset的长度必须保持一致喵~")
        processed_text = []
        processed_char_offsets = []
        for char,offset in zip(text,char_offsets):
            if self._is_chinese_char(char):
                processed_text.extend([" ", char, " "])
                processed_char_offsets.extend([(0,0), offset, (0,0)])
            else:
                processed_text.append(char)
                processed_char_offsets.append(offset)
        processed_text = "".join(processed_text)
        return processed_text, processed_char_offsets

    def _whitespace_tokenize_with_offsets(self, text, char_offsets):
        """
        空白切分方法，把含有空白字符串转换为列表
        之前的exampletokenizer给中文两侧加空格是为了使用split方便提出中文token，而这里只是让offset随着text同步变化，方便管理, 这里的offset已经变成了一个双层列表，每个子列表表示对应的token的offset
        :param text: 原始字符串
        :param char_offsets: 与text等长的offset列表
        :return: token_list，列表，进行空白切分后的列表, token_offsets，每个token对应的字符级offset，后续再得出每个token的offset
        """
        if not isinstance(text, str):
            raise ValueError(f"text的数据类型必须为字符串类型喵！")
        if not isinstance(char_offsets, list):
            raise ValueError(f"char_offsets的数据类型必须为列表类型喵！")
        if not len(text) == len(char_offsets):
            raise ValueError(f"text和char_offsets的长度不一致喵！请好好检查！")
        token_list = []
        token_offsets = []
        current_token = ""
        current_offsets = []
        for char, offset in zip(text,char_offsets):
            if char.isspace():
                if not current_token:
                    pass
                else:
                    token_list.append(current_token)
                    current_token = ""
                    token_offsets.append(current_offsets)
                    current_offsets = []
            else:
                current_token += char
                current_offsets.append(offset)
        if current_token:
            token_list.append(current_token)
            token_offsets.append(current_offsets)
        return token_list,token_offsets

    def _split_on_punctuation_with_offsets(self, token, char_offsets):
        """
        按字符切分字符串为token的列表，负责处理标点符号
        :param token: 输入的原始token
        :param char_offsets: 与token等长的offset列表
        :return: final_tokens: 处理标点符号后的token列表; final_offsets: 处理标点符号后，token的每个char对应的offset，(将标点符号单独提出为token)
        """
        if not isinstance(token, str):
            raise ValueError(f"输入的token必须要是字符串类型的喵！")
        if not isinstance(char_offsets, list):
            raise ValueError(f"输入的char_offsets必须是列表类型的喵！")
        if not len(token) == len(char_offsets):
            raise ValueError(f"输入的token和offsets必须是等长的喵！请好好检查！")
        tmp_tokens = []
        tmp_offsets = []
        final_tokens = []
        final_offsets = []
        for char, char_offset in zip(token, char_offsets):
            if self._is_punctuation(char):
                if len(tmp_tokens) > 0:
                    final_tokens.append("".join(tmp_tokens))
                    final_offsets.append(tmp_offsets)
                final_tokens.append(char)
                final_offsets.append([char_offset])
                tmp_tokens = []
                tmp_offsets = []
            else:
                tmp_tokens.append(char)
                tmp_offsets.append(char_offset)
        if len(tmp_tokens) > 0:
            final_tokens.append("".join(tmp_tokens))
            final_offsets.append(tmp_offsets)
        return final_tokens, final_offsets

    def _basic_tokenize_with_offsets(self, text):
        """
        完成基础的切分，返回token列表和token对应的offset列表
        :param text: 用户输入的原始字符串
        :return: final_token_list, final_token_offsets
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的内容必须是字符串类型的喵！")
        # 字符串清洗
        clean_text,char_offsets = self._clean_text_with_offsets(text)
        # 在汉字两侧添加空格
        clean_text,char_offsets = self._add_spaces_around_chinese_chars_with_offsets(clean_text,char_offsets)
        # 处理空格
        token_list, token_offsets = self._whitespace_tokenize_with_offsets(clean_text, char_offsets)
        final_token_list = []
        final_token_offsets = []
        for token, token_offset in zip(token_list, token_offsets):
            if self.do_lower_case:
                token = token.lower()
            if self.strip_accents or (self.strip_accents is None and self.do_lower_case):
                token = self._strip_accents(token)
            if len(token) != len(token_offset):
                raise ValueError(
                    f"token归一化后长度与offset数量不一致喵！"
                    f"token长度为{len(token)}，offset数量为{len(token_offset)}"
                )
            tokens, offsets = self._split_on_punctuation_with_offsets(token, token_offset)
            final_token_list.extend(tokens)
            final_token_offsets.extend(offsets)
        return final_token_list, final_token_offsets

    def _wordpiece_tokenize_with_offsets(self, token, char_offsets):
        """
        对单个token进行wordpiece切分, offset也转换成wordpiece形式
        :param token: 单个basic_token
        :param char_offsets: 单个token里面字符对应的offset列表
        :return: sub_tokens: list，进行wordpiece切分后的token list
        :return: sub_tokens_offsets: list 进行wordpiece切分后的字符级offset
        """
        if not isinstance(token, str):
            raise ValueError(f"传入的token必须为字符串类型喵")
        if not isinstance(char_offsets, list):
            raise ValueError(f"传入的char_offset必须为列表类型喵")
        if len(token) != len(char_offsets):
            raise ValueError(f"token的长度必须和char_offsets的长度一样喵")
        if token in self.token_to_id:
            return [token], [char_offsets]
        else:
            if len(token) > self.max_input_chars_per_word:
                return [self.unk_token], [char_offsets]
            else:
                start = 0  # 还未开始拆分时的起始位置
                sub_tokens = []  # 已经匹配成功的wordpiece
                sub_tokens_offsets = [] # 已经匹配成功的wordpiece对应的offset
                is_bad = False  # 是否出现无法匹配的部分
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
                            sub_tokens_offsets.append(char_offsets[start:end])
                            break
                        else:
                            end -= 1
                    if not current_sub_token:
                        is_bad = True
                        break
                    else:
                        start = end
                if is_bad:
                    return [self.unk_token], [char_offsets]
                else:
                    return sub_tokens, sub_tokens_offsets

    def tokenize_with_offsets(self, text):
        """
        tokenize主方法
        :param text: 用户输入的原始文本,str
        :return: final_token_list, list，tokenize的最终结果, merged_offsets, token的token级offset的列表
        """
        if not isinstance(text, str):
            raise ValueError(f"输入的text必须是字符串类型的喵！")
        basic_token_list, basic_token_offsets = self._basic_tokenize_with_offsets(text)
        final_token_list = []
        final_token_offsets = []
        for token, offset in zip(basic_token_list, basic_token_offsets):
            wordpiece_token_list, wordpiece_token_offsets = self._wordpiece_tokenize_with_offsets(token, offset)
            final_token_list.extend(wordpiece_token_list)
            final_token_offsets.extend(wordpiece_token_offsets)
        if len(final_token_list) != len(final_token_offsets):
            raise RuntimeError(
                f"WordPiece token数量和offset数量不一致喵！"
                f"token数量为{len(final_token_list)}，"
                f"offset数量为{len(final_token_offsets)}"
            )
        # 字符级offset转化为token级offset
        merged_offsets = []
        for char_offsets in final_token_offsets:
            token_offset = self._merge_char_offsets(char_offsets)
            merged_offsets.append(token_offset)
        return final_token_list, merged_offsets

    def _merge_char_offsets(self, char_offsets):
        """
        将字符级offset合并为标准的token级offset
        :param char_offsets: 单个token的字符级offset列表
        :return: token_offsets: 单个token的token级offset
        """
        if not isinstance(char_offsets, list):
            raise ValueError(f"char_offsets必须要是列表类型的喵！")
        if not len(char_offsets) > 0:
            raise ValueError(f"列表char_offsets的长度必须大于0喵！")
        if not all(isinstance(tup, tuple) for tup in char_offsets):
            raise ValueError(f"列表char_offsets里的元素都必须为元组喵！")
        if not all(len(tup) == 2 for tup in char_offsets):
            raise ValueError(f"列表char_offsets里的元组的长度都必须为2喵！")
        for tup in char_offsets:
            start, end = tup
            if not all(type(elem) is int for elem in tup):
                raise ValueError(f"列表char_offsets里的元组里的元素都必须为整型的喵！")
            if not end > start >= 0:
                raise ValueError(f"列表char_offsets里的元组里的元素都必须从小到大排列喵！必须都大于等于0喵！")
        for i in range(len(char_offsets) - 1):
            cur_tup = char_offsets[i]
            next_tup = char_offsets[i + 1]
            if cur_tup[-1] != next_tup[0]:
                raise ValueError(f"列表char_offsets里的元组不连续喵！cur_tup: {cur_tup}, next_tup: {next_tup}")
        token_offsets = (char_offsets[0][0], char_offsets[-1][1])
        return token_offsets

    def encode_with_offsets(self, text, add_special_tokens=True, truncation=False, padding=False, text_pair=None, max_length=None):
        """
                tokenizer的编码逻辑
                :param text: str, 源文本
                :param add_special_tokens: bool, 是否加入特殊token
                :param truncation: bool, 超过长度时是否进行截断
                :param padding: bool, 是否进行pad补位
                :param text_pair: None or str, 另一段文本
                :param max_length: int, token最大长度
                :return: encoded_set, dict {
                    'input_ids': input_ids, token转化为id后的列表
                    'token_type_ids': token_type_ids, token来自于哪个句子
                    'attention_mask': attention_mask, 1表示有效Token，0表示Padding}
                    'offset_mapping': offset_mapping, token列表对应的token级offset的列表
                """
        # 初始化最终输出
        encoded_dict = {}
        # 参数校验
        if not isinstance(text, str):
            raise ValueError(f"输入的text必须为字符串类型的喵！")
        if not isinstance(add_special_tokens, bool):
            raise ValueError(f"输入的add_special_tokens必须为bool类型的喵！")
        if not isinstance(truncation, bool):
            raise ValueError(f"输入的truncation必须为bool类型的喵！")
        if not isinstance(padding, bool):
            raise ValueError(f"输入的padding必须为bool类型的喵！")
        if not isinstance(text_pair, (str, type(None))):
            raise ValueError(f"输入的text_pair必须为字符串类型或者None类型的喵！")
        if type(max_length) is not int and max_length is not None:
            raise ValueError(f"输入的max_length必须为int类型或者None类型的喵！")
        if type(max_length) is int and max_length <= 0:
            raise ValueError(f"输入的max_length如果为int必须大于0喵！")
        if truncation and max_length is None:
            raise ValueError(f"说了要截断truncation就要告诉我多少才截断max_length喵！")
        if padding and (type(max_length) is not int or max_length <= 0):
            raise ValueError(f"如果要padding填充的话，max_length的值必须至少为正整数喵！")
        # 分词
        token_a,offsets_a = self.tokenize_with_offsets(text)
        if text_pair is not None:
            token_b, offsets_b = self.tokenize_with_offsets(text_pair)
            if add_special_tokens:
                if truncation:
                    special_tokens_count = 3
                    if max_length < special_tokens_count:
                        raise ValueError(f"max_length: {max_length} 的数值太小了喵！")
                    available_length = max_length - special_tokens_count
                    while len(token_a) + len(token_b) > available_length:
                        if len(token_a) > len(token_b):
                            token_a.pop()
                            offsets_a.pop()
                        else:
                            token_b.pop()
                            offsets_b.pop()
                token_list = [self.cls_token] + token_a + [self.sep_token] + token_b + [self.sep_token]
                token_type_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) + 1)
                offset_mapping = [(0,0)] + offsets_a + [(0,0)] + offsets_b + [(0,0)]
            else:
                if truncation:
                    special_tokens_count = 0
                    available_length = max_length - special_tokens_count
                    while len(token_a) + len(token_b) > available_length:
                        if len(token_a) > len(token_b):
                            token_a.pop()
                            offsets_a.pop()
                        else:
                            token_b.pop()
                            offsets_b.pop()
                token_list = token_a + token_b
                token_type_ids = [0] * len(token_a) + [1] * len(token_b)
                offset_mapping = offsets_a + offsets_b
        else:
            if add_special_tokens:
                if truncation:
                    special_tokens_count = 2
                    if max_length < special_tokens_count:
                        raise ValueError(f"max_length: {max_length} 的数值太小了喵！")
                    available_length = max_length - special_tokens_count
                    if len(token_a) > available_length:
                        token_a = token_a[:available_length]
                        offsets_a = offsets_a[:available_length]
                token_list = [self.cls_token] + token_a + [self.sep_token]
                token_type_ids = [0] * (len(token_a) + 2)
                offset_mapping = [(0,0)] + offsets_a + [(0,0)]
            else:
                if truncation:
                    special_tokens_count = 0
                    available_length = max_length - special_tokens_count
                    if len(token_a) > available_length:
                        token_a = token_a[:available_length]
                        offsets_a = offsets_a[:available_length]
                token_list = token_a
                token_type_ids = [0] * len(token_a)
                offset_mapping = offsets_a
        # 进一步组装参数
        input_ids = self.convert_tokens_to_ids(token_list)
        attention_mask = [1] * len(input_ids)
        if not (len(input_ids) == len(token_type_ids) == len(attention_mask) == len(offset_mapping)):
            raise RuntimeError(f"input_ids, token_type_ids, attention_mask, offset_mapping四个值的长度不一致喵！"
                               f"input_ids长度为：{len(input_ids)}"
                               f"token_type_ids长度为：{len(token_type_ids)}"
                               f"attention_mask长度为：{len(attention_mask)}"
                               f"offset_mapping长度为：{len(offset_mapping)}")
        if padding:
            padding_length = max_length - len(input_ids)
            if padding_length < 0:
                raise ValueError(f"目前input_ids的长度大于max_length，此时如果需要padding请开启truncation喵！")
            input_ids += [self.pad_token_id] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            offset_mapping += [(0,0)] * padding_length
            if not len(offset_mapping) == len(token_type_ids) == len(input_ids) == len(attention_mask) == max_length:
                raise ValueError(f"padding时出现异常！请检查源代码！")
        # 加入字典
        encoded_dict["input_ids"] = input_ids
        encoded_dict["token_type_ids"] = token_type_ids
        encoded_dict["attention_mask"] = attention_mask
        encoded_dict["offset_mapping"] = offset_mapping
        return encoded_dict