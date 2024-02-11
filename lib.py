from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
from functools import partial
#import autograd.numpy as np

def get_solution(X_train, y_train):
    LR = LogisticRegression(max_iter=1000000, tol=10e-8)
    LR.fit(X_train, y_train)
    return LR.coef_[0]

def W_full_base(n, x):
    return torch.Tensor(np.ones((n, n)))

def W_full(n):
    return partial(W_full_base, n)

def W_torus_base(n, x):
    side_length = int(n**0.5)
    
    while n % side_length != 0:
        side_length -= 1
    
    rows = side_length
    cols = n // rows
    
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        row, col = divmod(i, cols)
        
        top = ((row - 1) % rows) * cols + col
        bottom = ((row + 1) % rows) * cols + col
        left = row * cols + (col - 1) % cols
        right = row * cols + (col + 1) % cols

        matrix[i][top] = 1
        matrix[i][bottom] = 1
        matrix[i][left] = 1
        matrix[i][right] = 1
    return torch.Tensor(matrix) / np.sum(matrix[0])

def W_torus(n):
    return partial(W_torus_base, n)

def W_ring_base(n, x):
    if n == 1:
        return torch.ones([1, 1])
    elif n == 2:
        return torch.ones([2, 2]) / 2
    else:
        matrix = (np.eye(n, n, 1)+ np.eye(n, n, -1) + np.eye(n, n, 0)) / 3
        matrix[-1][0] = matrix[0][-1] = 1/3
        return torch.Tensor(matrix)

def W_ring(n):
    return partial(W_ring_base, n)

def W_centralized_base(n, x):
    return torch.Tensor(np.ones((n, n)) / n)

def W_centralized(n):
    return partial(W_centralized_base, n)

#k - number of local steps
def W_local_base(n, k, W_space, x):
    if (x + 1) % k != 0:
        return torch.eye(n)
    return torch.Tensor(W_space((x + 1)/k))

def W_local(n, k, W_space):
    return partial(W_local_base, n, k, W_space)

def W_id_base(n, x):
    return torch.eye(n)

def W_id(n):
    return partial(W_id_base, n)

def W_random(n):
    topologies = [W_torus(n), W_ring(n)]
    return topologies[random.randint(0, len(topologies) - 1)]

import random
from tqdm import tqdm
import time

def norm(x):
    return np.sqrt(np.sum(x**2))

def get_L(X):
    return np.mean(np.sum(X ** 2, axis=1))/4

def get_L_theor(X):
    return np.max(np.sum(X ** 2, axis=1)/4)

def get_L_v(X):
    return np.sum(X ** 2)/4

def gradient(X, y, w, lambd):
    return np.mean([(-y[i]*X[i]/(1+np.exp(y[i]*np.dot(w, X[i])))) for i in range(len(y))], axis=0)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import matplotlib.pyplot as plt
import autograd as ad
import autograd.numpy as np
from sklearn.model_selection import train_test_split   
from sklearn.datasets import load_svmlight_file
from random import Random

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        #rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(train_set, batch_size):
    size = dist.get_world_size()
    bsz = batch_size / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(train_set, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=int(bsz),
                                         shuffle=False)
    return train_set, bsz

def norm_torch(tensor):
    return torch.linalg.norm(tensor)

# If data is given in Subset-like shape we need to split it to features and targets
def get_solution_torch(dataset):
    data = [entry[0] for entry in dataset]
    targets = [entry[1][0] for entry in dataset]
    return get_solution(data, targets)
    
def get_grad_norm_torch(model, dataset, error):
    norm = 0.
    world_size = dist.get_world_size()
    model.eval()
    
    targets = torch.stack([entry[0] for entry in dataset])
    labels = torch.stack([entry[1] for entry in dataset])
    
    for param in model.parameters():
        model.zero_grad()
        output = model(targets)
        loss = error(output, labels)
        loss.backward()
        
        tensor_list = [param.grad.data.clone() for _ in range(world_size)]
        dist.all_gather(tensor_list, param.grad.data)
        tensor = torch.sum(torch.stack(tensor_list), dim=0) / world_size
        norm += norm_torch(tensor)
        break # iterate through all parameters but last, there has to be more elegant way
    model.zero_grad()
    model.train()
    return norm

def train(model, device, dataset, batch_size, W_k, num_epochs=1000, lr=1/5.25, display_interval=10):
    error = nn.SoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    norm0 = get_grad_norm_torch(model, dataset, error)
    train_loader, bsz = partition_dataset(dataset, batch_size)
    solution = torch.tensor(get_solution_torch(dataset)).reshape(1, -1)
    rank = dist.get_rank()
    curr_iter = 0
    errors = []
    model.train()
    for epoch in tqdm(range(num_epochs)):
        loss_epoch = 0
        for idx, (batch, labels) in enumerate(train_loader):
            curr_iter += 1
            optimizer.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            loss = error(output, labels)
            loss_epoch += loss
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                tensor_list = [param.data.clone() for _ in range(dist.get_world_size())] #tensors of size (1, #number of features)
                dist.all_gather(tensor_list, param.data)
                #print(curr_iter, W_k(curr_iter))
                param.data = (W_k(curr_iter).T[rank] @ torch.cat(tensor_list, dim=0)).reshape(1, -1)
            if curr_iter % 100 == 0:
                norm1 = get_grad_norm_torch(model, dataset, error)
                model.eval()
                norm2 = 0.
                for param in model.parameters():
                    tensor_list = [param.data.clone() for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list, param.data)
                    tensor_list = [tensor - solution for tensor in tensor_list]
                    norm2 += np.mean([norm_torch(tensor_list[i]) ** 2 for i in range(len(tensor_list))])
                    break
                model.train()
                errors.append([curr_iter, norm1 / norm0, norm2])
#        if epoch % display_interval == 0:
#            print('Rank: {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.10f}'.format(
#                rank, epoch, idx * len(batch), len(train_loader.dataset),
#                100 * idx / len(train_loader), loss.item()))
    return np.transpose(errors)

def test(model, device, test_loader):
    error = nn.SoftMarginLoss(reduction='sum')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = error(output, labels)
            test_loss += loss.item()
            #pred = output.max(1)[1]
            pred = (output>0.5)*2-1
            correct += pred.eq(labels).sum().item()
    test_loss /= (len(test_loader.dataset) * 10)
    #print('Test set: Average Loss {:.6f}, Accuracy: {:.2f}%'.format(
    #    test_loss, correct / len(test_loader.dataset) * 100))

class LinearRegressionModel(nn.Module):
    def __init__(self):
        input_dim = 112
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        logits = self.linear(x)
        return logits

class NN():
    def __init__(self, input_dim):
        super().__init__()

    def forward(self, x):
        output = 1
        return output

def worker_process(backend, errors, rank, world_size, num_epochs, batch_size, W, lr=1/5.25):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29510'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    dataset = "mushrooms.txt"
    data = load_svmlight_file(dataset)
    X, y = data[0].toarray(), data[1]
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    y = 2 * y - 3
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_set = [[x, y] for x, y in zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).reshape([-1, 1]))]
    test_set = [[x, y] for x, y in zip(torch.from_numpy(X_test), torch.from_numpy(y_test).reshape([-1, 1]))]
    
    device = torch.device('cpu')
    errors_w = []
    model = LinearRegressionModel()
    #model = model.double()
    model.to(device)
    train_sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    errors_w = train(model, device, train_set, batch_size, W, num_epochs=num_epochs, lr=lr)
    if dist.get_rank() == 0:
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)
        test(model, device, test_loader)
        errors.put(errors_w)
    dist.destroy_process_group()
    del os.environ['MASTER_ADDR']
    del os.environ['MASTER_PORT']

def init_DSGD(errors, world_size, num_epochs, batch_size, W, lr=1/5.25):
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_process, args=('gloo', errors, rank, world_size, num_epochs, batch_size, W, lr))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
