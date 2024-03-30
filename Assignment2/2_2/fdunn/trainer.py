import os
import pickle
import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optimizer, loss_fn, epoch=6):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = epoch

        self.valid_scores = []

        self.train_loss = []
        self.valid_loss = []

    def train(self, train_loader, test_loader):
        for i in range(self.epoch):
            print("epoch:", i)
            train_loss_value = 0.0
            # 开始分批次加载图片和标签
            for images, labels in tqdm(train_loader):
                output = self.model(images)
                # 累积损失函数值并进行反向传播
                train_loss_value += self.loss_fn(output, labels)[0]
                self.loss_fn.backward()
                self.optimizer.step()

            train_loss_value /= len(train_loader)
            self.train_loss.append(train_loss_value)

            valid_loss_value = 0
            valid_correct = 0
            # 分批次加载验证图片和标签
            for images, labels in test_loader:
                output = self.model(images)
                loss, probability_pred = self.loss_fn(output, labels)
                valid_correct += np.sum((np.argmax(probability_pred, axis=1) == labels.numpy()))
                valid_loss_value += loss

            valid_loss_value /= len(test_loader)
            print('loss: ', valid_loss_value)
            self.valid_loss.append(valid_loss_value)

            valid_accuracy_value = valid_correct / len(test_loader.dataset)
            print('accuracy: ', valid_accuracy_value)
            self.valid_scores.append(valid_accuracy_value)

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        self.dev_loss.append(loss)
        score = self.metric(logits, y)
        self.dev_scores.append(score)
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_dir):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                with open(os.path.join(save_dir, layer.name+".pdparams"),'wb') as fout:
                    pickle.dump(layer.params, fout)

    def load_model(self, model_dir):
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams","")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                with open(file_path,'rb') as fin:
                    layer.params = pickle.load(fin)