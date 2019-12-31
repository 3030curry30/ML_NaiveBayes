import codecs
import jieba
import re


def get_stop_word(stop_path):
    # open stop word file
    stop_word = {}
    with codecs.open(stop_path, "r", encoding='utf8', errors='ignore') as f:
        for index, line in enumerate(f.readlines()):
            word = line.rstrip('\n').rstrip('\r')

            if word not in stop_word:
                stop_word[word] = index
    return stop_word


# a file path get data like: [[], [], []]
# length is the number of texts
def get_data(file_path, stop_word):
    data = []
    with codecs.open(file_path, "r", encoding='gb18030', errors='ignore') as f:
        file_data = f.read()
        pattern = r'<text>([\s\S]*?)</text>'

        # get each document
        res = re.findall(pattern, file_data)

        # drop some symbol
        for index in range(len(res)):
            res[index] = res[index].replace(' ', '')
            res[index] = res[index].replace(u'\u3000', u'')
            res[index] = res[index].replace('\r\n', '')

        for index in range(len(res)):
            word_list = [x for x in jieba.cut(res[index]) if x not in stop_word]
            data.append(word_list)

    return data


def get_vocab(train_data, test_data):
    # preserve 5 index to other
    vocab, index = {"<UNK>": 0}, 1

    # get vocab by train data
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            for word in train_data[i][j]:
                if word not in vocab:
                    vocab[word] = index
                    index += 1

    # get vocab by test data
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            for word in test_data[i][j]:
                if word not in vocab:
                    vocab[word] = index
                    index += 1

    return vocab


if __name__ == '__main__':
    stop_word = get_stop_word('./data/stop_words_zh.txt')
    data = get_data('./data/Tsinghua/train/体育', stop_word)
