import numpy as np
import matplotlib.pyplot as plt


class SoftMax(object):
    def __init__(self, train_data, test_data, vocab, representation_type):
        self.vocab = vocab
        self.num_classes = len(train_data)
        self.representation = representation_type

        # get train data and test data
        self.train_data, self.train_target = self.get_BOW(train_data)
        self.test_data, self.test_target = self.get_BOW(test_data)

        # data: [n, len(vocab)], target: [n, num_classes]
        # w: [len(vocab), num_classes]
        # self.w = np.zeros((len(vocab), self.num_classes), dtype=float)
        self.w = np.ones((len(vocab), self.num_classes), dtype=float)
        # self.w = np.random.rand(len(vocab), self.num_classes)

        self.figure, self.ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        self.loss_col, self.epoch_col, self.Acc_col, self.Acc_epoch = [], [], [], []

    # get the text of bag of word embedding
    def get_BOW(self, data):
        final_data, target = [], []

        # i denote each class, j denote each text
        for i in range(len(data)):
            for j in range(len(data[i])):
                bow_embedding = [0] * len(self.vocab)
                word_dict = set()

                for word in data[i][j]:
                    if word in word_dict:
                        continue

                    if word in self.vocab and word not in word_dict:
                        bow_embedding[self.vocab[word]] += 1

                    # unknown vocab
                    elif word not in self.vocab:
                        bow_embedding[0] += 1

                    # using presence BOW method
                    if self.representation == 0:
                        word_dict.add(word)

                final_data.append(bow_embedding)
                target.append(i)
        return np.array(final_data), np.array(target)

    def Gradient_Descent(self, epochs, exp, lr=0.1):
        cost = 0
        for epoch in range(epochs):
            z = np.dot(self.train_data, self.w)
            y_hat = self.softmax_function(z)

            cur_cost = self.loss(self.train_target, y_hat)
            self.w += lr * self.loss_back(self.train_target, y_hat)

            self.loss_col.append(cur_cost)
            self.epoch_col.append(epoch)
            self.dy_plot()
            if (epoch % 20 == 0 and epoch >= 20) or epoch == 0 or epoch == epochs - 1:
                self.Acc_epoch.append(epoch)
                self.prediction()

            if abs(cur_cost - cost) < exp:
                plt.pause(10)
                plt.close()
                break
            else:
                cost = cur_cost
        plt.pause(10)
        plt.close()

    def softmax_function(self, z):
        mmax = np.max(z, axis=1).reshape(-1, 1)
        temp = np.exp(z - mmax)
        return temp / temp.sum(axis=1, keepdims=True)

    def loss(self, y, y_hat):
        loss = 0
        for i in range(np.size(y)):
            loss += -np.log(y_hat[i, y[i]])
        return loss

    def loss_back(self, y, y_hat):
        y_hat = -y_hat
        for i in range(np.size(y)):
            y_hat[i, y[i]] += 1

        w_grad = 1 / len(self.train_data) * np.dot(self.train_data.T, y_hat)
        return w_grad

    def prediction(self):
        n = np.size(self.test_target)
        pred = np.argmax(np.dot(self.test_data, self.w), axis=1)
        correct = np.sum(pred == self.test_target)
        Acc = correct / n * 100
        self.Acc_col.append(Acc)

        self.ax[1].cla()
        self.ax[1].set_title("Accuracy Plot")
        self.ax[1].set_ylabel("accuracy")
        self.ax[1].set_xlabel("epoch")
        self.ax[1].scatter(self.Acc_epoch, self.Acc_col, color="red")
        print("Accuracy: {}%, correct/total: {}/{}".format(Acc, correct, n))

    def dy_plot(self):
        # clean plot
        self.ax[0].cla()

        # draw loss and Acc plot
        title = "presence method BOW" if self.representation == 0 else "TF method BOW"
        self.figure.suptitle("{} loss and accuracy plot".format(title))
        self.ax[0].set_title("Loss Plot")
        self.ax[0].set_ylabel("loss")
        self.ax[0].set_xlabel("epoch")

        self.ax[0].scatter(self.epoch_col, self.loss_col, color="red")
        plt.pause(0.1)