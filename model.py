import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from data.graph.layer import GraphAttentionLayer

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)

        out += self.bias
        out = self.relu(out)

        return out, A





class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.num_head = 4

        self.layers = nn.Sequential(
            GCN(),
            GCN(),
        )

        self.proj = nn.Sequential(
            nn.Linear(26000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.att = GraphAttentionLayer()



    def forward(self, X, A):
        # GCN
        # out = self.layers((X, A))[0]

        # GAT
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            # 2层gat
            for _ in range(self.num_head):
                x = self.att(x, a)
                feature_temp.append(x)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)

        return out

class FpModel(nn.Module):
    def __init__(self):
        super(FpModel, self).__init__()

        self.fp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp1 = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fp3 = nn.Sequential(
            nn.Linear(780, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)

        # self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, x1, x2, x3, x4):
        x = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)

        return x, x1, x2, x3, x4

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.graph = GraphModel()
        self.fp = FpModel()
        self.proj = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )
        self.fc = nn.Linear(128, 1)
        self.active = nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, f, f1, f2, f3, f4, X, A, label):
        # f, f1, f2, f3, f4 = self.fp(f, f1, f2, f3, f4)
        X = self.graph(X, A)

        # x = torch.cat((f, f1, f3, X), dim=1)
        # x = self.proj(x)
        x = self.fc(X)
        x = self.active(x).squeeze(-1)
        loss = self.loss_fn(x, label)

        return x, loss


if __name__ == '__main__':
    from data.graph.utils import load_data, MyDataset, set_seed, load_fp
    from torch.utils.data import DataLoader
    import torch
    import pandas as pd

    SEED = 42
    set_seed(SEED)


    path = r'C:\Users\LRS\Desktop\wty\WTY\data\Smiles.csv'
    PubchemFP881_path = r'C:\Users\LRS\Desktop\wty\WTY\data\PubchemFP881.csv'
    GraphFP1024_path = r'C:\Users\LRS\Desktop\wty\WTY\data\GraphFP1024.csv'
    APC2D780_path = r'C:\Users\LRS\Desktop\wty\WTY\data\APC2D780.csv'
    FP1024_path = r'C:\Users\LRS\Desktop\wty\WTY\data\FP1024.csv'
    X, A, mogen_fp, labels = load_data(path)

    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(GraphFP1024_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(FP1024_path)


    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)
    labels = torch.FloatTensor(labels)

    train_dataset = MyDataset(f=mogen_fp[0:8000], f1=fp1[0:8000], f2=fp2[0:8000], f3=fp3[0:8000], f4=fp4[0:8000], X=X[0:8000], A=A[0:8000], label=labels[0:8000])
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=True)
    test_dataset = MyDataset(f=mogen_fp[8000:], f1=fp1[8000:],  f2=fp2[8000:], f3=fp3[8000:], f4=fp4[8000:], X=X[8000:], A=A[8000:], label=labels[8000:])
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, drop_last=True)

    print(X.shape, A.shape)
    # graph = GraphModel()
    # model = FpModel()

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)


    from sklearn.metrics import roc_auc_score, accuracy_score
    model.train()
    for epoch in range(20):
        pred_y = []
        PED = []
        ture_y = []
        for i, batch in enumerate(train_loader):
            fp, fp1, fp2, fp3, fp4, X, A, label = batch
            optimizer.zero_grad()
            logits, loss = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            # logits, loss = model(fp, label)
            loss.backward()
            optimizer.step()
            lr_optimizer.step()
            print('Epoch:', epoch, 'Loss:', loss.item())

            # logits = torch.argmax(logits, dim=1)
            logits = logits.detach().numpy()
            pred_y.extend(logits)

            PED.extend(logits.round())

            label = label.numpy()
            ture_y.extend(label)

        if epoch == 19:
            acc = accuracy_score(ture_y, PED)
            auc = roc_auc_score(ture_y, pred_y)
            print('训练集准确率ACC:', acc)
            print('训练集AUC:', auc)

    model.eval()
    pred_y = []
    PED = []
    ture_y = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            fp, fp1, fp2, fp3, fp4, X, A, label = batch

            logits, loss = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            # logits, loss = model(fp, label)
            print('Loss:', loss.item())

            # logits = torch.argmax(logits, dim=1)
            logits = logits.detach().numpy()
            pred_y.extend(logits)

            PED.extend(logits.round())
            label = label.numpy()
            ture_y.extend(label)

        acc = accuracy_score(ture_y, PED)
        auc = roc_auc_score(ture_y, pred_y)
        print('测试集准确率ACC:', acc)
        print('测试集AUC:', auc)


