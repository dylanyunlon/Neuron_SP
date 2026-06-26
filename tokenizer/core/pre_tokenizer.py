"""
GPT-2 风格预分词器 (Pre-Tokenizer)

使用正则表达式将文本按空格/标点/换行切分成 word-level chunks。
BPE merge 在每个 chunk 内独立学习，与 GPT-2 原始实现保持一致。

参考：https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import re
from typing import List

# GPT-2 原始正则模式
# 匹配优先级（从高到低）：
#   1. 英语缩写后缀：'s  't  're  've  'm  'll  'd
#   2. 可选前导空格 + 单词字符序列（字母/数字/下划线）
#   3. 可选前导空格 + 非空白非单词字符序列（标点/符号等）
#   4. 纯空白（后面没有非空白字符，即行尾/段落末尾的空白）
#   5. 其余空白（空格、制表符、换行等）
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class PreTokenizer:
    """
    GPT-2 风格预分词器。

    将原始文本切分为 word-level chunks，后续 BPE 算法在每个
    chunk 内部独立执行 merge，从而避免跨词边界合并字节对。

    示例
    ----
    >>> pt = PreTokenizer()
    >>> pt.pre_tokenize("Hello, world! I'm fine.")
    ['Hello', ',', ' world', '!', ' I', "'m", ' fine', '.']
    """

    def pre_tokenize(self, text: str) -> List[str]:
        """
        对输入文本执行预分词。

        Parameters
        ----------
        text : str
            待分词的原始文本字符串。

        Returns
        -------
        List[str]
            切分后的 token 列表（word-level chunks）。
        """
        return GPT2_PAT.findall(text)


# ---------------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pt = PreTokenizer()

    test_cases = [
        "Hello, world!",
        "I'm not sure what you're doing.",
        "She said \"I'll be there\".",
        "  leading spaces and\nnewlines\t here  ",
        "café résumé naïve",
        "2024-01-01 price: $99.99",
    ]

    for text in test_cases:
        tokens = pt.pre_tokenize(text)
        print(f"Input : {repr(text)}")
        print(f"Output: {tokens}")
        print()
