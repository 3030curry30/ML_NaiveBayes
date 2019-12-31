import argparse
import DataProcess
import SoftMax
import os
import Bernoulli
import Multinomial
import numpy as np


# input train and test file path
def main():
    # set default file path
    default_train = './data/Tsinghua/train'
    default_test = './data/Tsinghua/test'
    default_stop = './data/stop_words_zh.txt'

    # input some parameters
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'
    parser.add_argument('--train', help='train data file path', default=default_train)
    parser.add_argument('--test', help='test data file path', default=default_test)
    parser.add_argument('--stop', help='stop word data file path', default=default_stop)
    parser.add_argument('--alpha', help='Multinomial smoothing parameter', default=0.9)
    parser.add_argument('--beta', help='Bernoulli smoothing parameter', default=0.1)

    args = parser.parse_args()

    # judge file path if exist
    if not os.path.exists(args.train) or not os.path.exists(args.test):
        print("Error file path!")
        exit(0)

    # train path and test path
    train_list = os.listdir(args.train)
    test_list = os.listdir(args.test)

    # get stop word
    stop_word = DataProcess.get_stop_word(args.stop)

    # get train data
    train_data, test_data = [], []
    for name in train_list:
        path = args.train + "/" + name
        train_data.append(DataProcess.get_data(path, stop_word))

    for name in test_list:
        path = args.test + "/" + name
        test_data.append(DataProcess.get_data(path, stop_word))

    # get vocab to BOW model
    vocab = DataProcess.get_vocab(train_data, test_data)

    # # define softmax model, 0: denote presence and 1 denote frequency
    # model = SoftMax.SoftMax(train_data, test_data, vocab, 0)
    # model.Gradient_Descent(100, 1e-7, 0.5)

    # Bernoulli Naive Bayes Model
    alpha = args.beta
    model = Bernoulli.BernoulliNB(train_data, test_data, vocab, alpha)
    model.prediction()

    # Multinomial Naive Bayes Model
    alpha = args.alpha
    model = Multinomial.MultinomialNB(train_data, test_data, vocab, alpha)
    model.prediction()


if __name__ == "__main__":
    main()