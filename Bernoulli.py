import numpy as np


class BernoulliNB(object):
    def __init__(self, train, test, vocab, alpha=1):
        self.n = 0
        self.vocab = vocab
        self.alpha = alpha
        self.vocab_size = len(vocab)
        self.n_classes = len(train)
        self.n_features = len(vocab)

        self.train_data, self.train_target = self.get_standard_data(train)
        self.test_data, self.test_target = self.get_standard_data(test)

        self.class_prob = np.zeros((1, self.n_classes), dtype=float)
        self.features_prob = np.zeros((self.n_classes, self.n_features), dtype=float)

    # list to numpy
    def get_standard_data(self, data):
        final_data, label = [], []

        for c in range(len(data)):
            for i in range(len(data[c])):
                temp = [0] * self.n_features
                for word in data[c][i]:
                    temp[self.vocab[word]] = 1
                final_data.append(temp)
                label.append(c)
        return np.array(final_data), np.array(label)

    def calculate_prob(self):
        # calculate class probability
        text_count = np.size(self.train_target)
        for c in range(self.n_classes):
            self.class_prob[0, c] = np.sum(self.train_target == c)

        # smoothing
        self.class_prob = np.log(self.class_prob + 1) - np.log(text_count + self.n_classes)

        # calculate feature probability
        for c in range(self.n_classes):
            mask = (self.train_target == c)
            text_count = np.sum(mask)

            # feature_count.shape = [1, n_features]
            # the number of feature_count in each class
            feature_count = np.dot(mask.T, self.train_data)
            self.features_prob[c, :] = np.log(feature_count + self.alpha) - np.log(text_count + 2 * self.alpha)

    def prediction(self):
        self.calculate_prob()
        neg_feature_prob = np.log(1 - np.exp(self.features_prob))
        neg_feature = (self.test_data == 0)

        pred_prob = self.class_prob + np.dot(neg_feature, neg_feature_prob.T) + \
               np.dot(self.test_data, self.features_prob.T)

        y_hat = np.argmax(pred_prob, axis=1)
        print("Bernoulli Naive Bayes Acc: {:.2f}%, alpha: {:.4f}"
              .format(100 * np.sum(y_hat == self.test_target) / np.size(self.test_target), self.alpha))