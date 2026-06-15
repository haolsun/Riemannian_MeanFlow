import torch.nn as nn
import torch
from gfm.data.components.util import cartesian_from_latlon
from csv import reader
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import os

class Classifier(nn.Module):
    def __init__(self, model_in=3, model_out=4, model_hidden=256):
        super(Classifier, self).__init__()
        self.model_in = model_in
        self.model_out = model_out
        self.model_hidden = model_hidden

        self.linear1 = nn.Linear(model_in, model_hidden)
        self.linear2 = nn.Linear(model_hidden, model_hidden)
        self.linear3 = nn.Linear(model_hidden, model_out)

    def forward(self, x):
        layer_1 = torch.relu(self.linear1(x))
        layer_2 = torch.relu(self.linear2(layer_1))
        layer_3 = torch.softmax(self.linear3(layer_2), dim=1)
        return layer_3

def load_top(data, amino, label):
    data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
    data = torch.tensor(data % 360 * torch.pi / 180)
    # print('no label:',data.shape)
    data = torch.cat([data, torch.full([data.shape[0],1], label)], dim=1)
    # print('have label:', data.shape)
    return data


def data_load():
    data = pd.read_csv(
        "data/top500/aggregated_angles.tsv",
        delimiter="\t",
        names=["source", "phi", "psi", "amino"],
    )
    data_1 = load_top(data, 'General', 1.)
    data_2 = load_top(data, 'Glycine', 2.)
    data_2 = torch.cat([data_2 for i in range(10)], dim=0)
    data_3 = load_top(data, 'Proline', 3.)
    data_3 = torch.cat([data_3 for i in range(20)], dim=0)
    data_4 = load_top(data, 'Pre-Pro', 4.)
    data_4 = torch.cat([data_4 for i in range(20)], dim=0)
    data = torch.cat((data_1, data_2, data_3, data_4), dim=0)
    data = data[torch.randperm(data.size(0))]
    return data


def load_so3():
    dirname = '/data1/zzc/code/gfm/data/raw'
    filename = os.path.join(dirname, 'cone_train.npy')
    data_1 = np.load(filename).astype("float32")
    data_1 = data_1[:20000]
    data_1 = torch.from_numpy(data_1).float()
    data_1 = data_1.view(-1, 9)
    data_1 = torch.cat([data_1, torch.full([data_1.shape[0], 1], 1)], dim=1)

    filename = os.path.join(dirname, 'fisher24_train.npy')
    data_2 = np.load(filename).astype("float32")
    data_2 = data_2[:20000]
    data_2 = torch.from_numpy(data_2).float()
    data_2 = data_2.view(-1, 9)
    data_2 = torch.cat([data_2, torch.full([data_2.shape[0], 1], 2)], dim=1)

    N = 20000
    noise = 0.1
    proj_y = 10.
    generator = torch.Generator()
    generator.manual_seed(42)
    t = 3 * np.pi * (1 - torch.rand(N, generator=generator))
    x = t * torch.cos(t)
    z = t * torch.sin(t)
    x += noise * torch.randn(N, dtype=torch.float, generator=generator)
    z += noise * torch.randn(N, dtype=torch.float, generator=generator)

    target = F.normalize(torch.stack([x, torch.ones_like(x) * proj_y, z], dim=1), dim=1)
    source = torch.tensor([0, 1, 0], dtype=torch.float).unsqueeze(0)
    axis = torch.cross(source, target, dim=1)
    theta = torch.acos(torch.clamp(torch.sum(source * target, dim=1), -1.0, 1.0))
    dataset = F.normalize(axis, dim=-1) * theta.unsqueeze(1)
    dataset = Rotation.from_rotvec(dataset).as_matrix()
    # print('dataset: ',dataset.shape)
    data_3 = torch.Tensor(dataset).float()
    data_3 = data_3.view(-1,9)
    data_3 = torch.cat([data_3, torch.full([data_3.shape[0], 1], 3)], dim=1)
    data = torch.cat((data_1, data_2, data_3), dim=0)
    data = data[torch.randperm(data.size(0))]
    return data


def correct_and_incorrect(y, label):
    correct = 0
    fail = []  # 把应该预测的预测成了其他  如 1-》2
    output = model(y)
    for i in range(label.shape[0]):
        max_poss = 0
        max_position = 1
        for j in range(3):
            if output[i][j] > max_poss:
                max_poss = output[i][j]
                max_position = j+1
        # print('预测矩阵：',output[i],'max_position',max_position,'label:',label[i][0])
        if max_position == label[i][0]:
            correct += 1
        else:
            fail.append([label[i][0], max_position])
    print('correct:', correct/label.shape[0])
    # print('fail:', fail)


data = load_so3()
data_len = data.shape[0]
train_data = data[:int(0.8 * data_len)]
val_data = data[int(0.8 * data_len):int(0.9 * data_len)]
test_data = data[int(0.9 * data_len):]

model = Classifier(model_in=9, model_out=3, model_hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
x, label = torch.split(train_data, [9, 1], dim=1)
y, label_y = torch.split(val_data, [9, 1], dim=1)
z, label_z = torch.split(test_data, [9, 1], dim=1)
label_use = []
for i in range(label.shape[0]):
    if label[i][0] == 1:
        label_use.append([1., 0., 0.])
    if label[i][0] == 2:
        label_use.append([0., 1., 0.])
    if label[i][0] == 3:
        label_use.append([0., 0., 1.])
label_use = torch.Tensor(label_use)
# print(x.shape)
# print(label_use.shape)
for epoch in range(500):
    # print('epoch:', epoch, flush=True)
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, label_use)
    print('loss:', loss)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('val start.....')
        correct_and_incorrect(y, label_y)
        print('val end.....')


print('test start.....')
correct_and_incorrect(z, label_z)




