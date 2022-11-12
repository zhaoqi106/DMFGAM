import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from graph.layer import GraphAttentionLayer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


if __name__ == '__main__':
    from graph.utils import load_data, MyDataset, set_seed, load_fp
    from torch.utils.data import DataLoader
    import torch
    import pandas as pd

    SEED = 42
    set_seed(SEED)

    path = r'D:\科研\王天一程序\data\newcadata.csv'
    X, A, mogen_fp, labels = load_data(path)



    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    labels = torch.FloatTensor(labels)

    from sklearn.ensemble import RandomForestClassifier

    # 构建模型，用决策树和随机森林进行对比
    model = RandomForestClassifier(random_state=0)

    model.fit(mogen_fp[0:12620], labels[0:12620])

    # score_r = model.score(mogen_fp[8000:10355], labels[8000:10355])

    pred = model.predict(mogen_fp[-44:])

    TN, FP, FN, TP = confusion_matrix(labels[-44:], pred).ravel()
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)

    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    print('TN, FP, FN, TP:', TN, FP, FN, TP)
    print('SPE, SEN, NPV, PPV, MCC:', SPE, SEN, NPV, PPV, MCC)
    acc = accuracy_score(labels[-44:], pred)
    auc = roc_auc_score(labels[-44:], pred)
    print('测试集准确率ACC:', acc)
    print('测试集AUC:', auc)


