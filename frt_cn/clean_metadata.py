import os
import re

from g2pc import G2pC
from tqdm import tqdm
import random


def init_dict():
    char2index = {}
    dict_path = os.path.dirname(__file__)

    with open(os.path.join(dict_path, "char_dict.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            char, index = line.split(" ")
            char2index[char] = index
    return char2index


def custom_mandarine_cleaners(content):
    cn_punctuation = "！‘’“”《》，：；、。？"
    en_punctuation = "!'\",:;.?"
    cn2en_trans = '!\'\'"""",:;,.?'
    en2dict_trans = "!'',,,.?"

    # 定义从原始句子内容中直接剔除的字符集（这里只保留汉字和中英文标点）
    # 匹配一个或者多个特殊字符，^代表非。即匹配字符串中的所有“非”以下类别的字符：
    # \u4e00-\u9fa 汉字字符
    # 汉字标点符号
    # 英文标点符号
    del_symbols = re.compile(
        r"[^\u4e00-\u9fa5%s%s]+" % (cn_punctuation, en_punctuation)
    )

    # 匹配不可见字符集
    unseen_symbols = re.compile(r"[\s]+")

    # 清除不需要的字符
    content = del_symbols.sub("", content)
    content = unseen_symbols.sub(" ", content)

    # 将所有中文标点转换为英文标点，再将得到的所有英文标点转换为字典内标点
    trans_tab = str.maketrans(cn_punctuation, cn2en_trans)
    content = content.translate(trans_tab)
    trans_tab = str.maketrans(en_punctuation, en2dict_trans)
    content = content.translate(trans_tab)
    return content


def preprocess_biaobei(metadata):
    with open(metadata, "r", encoding="utf-8") as f:
        lines = [line for line_id, line in enumerate(f) if line_id % 2 == 0]

    clean_texts = []
    for line in tqdm(lines, desc="cleaning texts: "):
        utterence_id, content = line.split("\t")
        content = content.strip()
        content = custom_mandarine_cleaners(content)
        clean_texts.append((utterence_id, content))
    return clean_texts


def clean_biaobei(metadata, char2index):
    g2pc = G2pC()
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")

    clean_texts = preprocess_biaobei(metadata)
    f_write = open(os.path.join(filelists_path, "data.csv"), "w", encoding="utf-8")
    for text in tqdm(clean_texts, desc="normalizing the texts: "):
        utterence_id, content = text
        clean_words = g2pc(content)

        # 将g2pc分词后的拼音连接起来, 单独处理v这个拼音, 同时去掉分词内空格
        clean_words = [word[3].replace("u:", "v") for word in clean_words]
        clean_words = [word.replace(" ", "") for word in clean_words]
        clean_char = " ".join(clean_words)
        clean_char = clean_char.upper()

        # 按照音频id, 原句（使用字典内英文标点符号），token，token_id方式写入
        normalized_char = []
        token_id = []
        for char in clean_char:
            if char in char2index.keys():
                normalized_char.append(char)
                token_id.append(char2index[char])
            elif char == " ":
                normalized_char.append("<space>")
                token_id.append(char2index["<space>"])
            else:
                normalized_char.append("<unk>")
                token_id.append(char2index["<unk>"])
        normalized_char.append("<eos>")
        token_id.append(char2index["<eos>"])

        normalized_char = " ".join(normalized_char)
        token_id = " ".join(token_id)
        f_write.write(
            utterence_id + "|" + content + "|" + normalized_char + "|" + token_id + "\n"
        )
    f_write.close()


def make_subsets():
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")
    data_path = os.path.join(filelists_path, "data.csv")
    train_path = os.path.join(filelists_path, "train_set.csv")
    dev_path = os.path.join(filelists_path, "dev_set.csv")
    test_path = os.path.join(filelists_path, "test_set.csv")
    long_path = os.path.join(filelists_path, "long_set.csv")

    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line for line in f]

    lines = sorted(lines, key=lambda x: len(x.split("|")[-1]))
    lines_domain = lines[:-100]
    lines_outdom = lines[-100:]

    random.seed(0)
    random.shuffle(lines_domain)
    train_lines = lines_domain[:-500]
    dev_lines = lines_domain[-500:-250]
    test_lines = lines_domain[-250:]
    long_lines = lines_outdom

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)

    with open(dev_path, "w", encoding="utf-8") as f:
        for line in dev_lines:
            f.write(line)

    with open(test_path, "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line)

    with open(long_path, "w", encoding="utf-8") as f:
        for line in long_lines:
            f.write(line)
    print("All the subsets have been prepared")


if __name__ == "__main__":
    # metadata path
    metadata = "/home/server/disk1/DATA/BIAOBEI/BIAOBEI-Corpus/000001-010000.txt"

    # create the char dict
    char2index = init_dict()

    # create the data.csv in filelists
    clean_biaobei(metadata, char2index)

    # split the data.csv into train, dev, and test
    make_subsets()
