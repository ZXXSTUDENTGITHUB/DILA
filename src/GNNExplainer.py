import random

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import os
import numpy as np
from torch.nn import BatchNorm1d
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.metric import fidelity
from torch_geometric.explain.metric import fidelity_curve_auc
from torch_geometric.explain.metric import groundtruth_metrics
from tqdm import tqdm

from utils import *
from datetime import datetime, timedelta

current_time = datetime.now()
print("Current time:", current_time)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.conv3 = GCNConv(hidden_dim, int(output_dim))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))

        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


def load_XA(dataname, datadir="XAL"):
    prefix = os.path.join(datadir, dataname)
    filename_A = prefix + "_A.npy"
    filename_X = prefix + "_X.npy"
    A = np.load(filename_A)
    X = np.load(filename_X)
    return A, X


def load_labels(dataname, datadir="XAL"):
    prefix = os.path.join(datadir, dataname)
    filename_L = prefix + "_L.npy"
    L = np.load(filename_L)
    return L


def load_ckpt(dataname, datadir="XAL", isbest=False):
    """Load a pre-trained pytorch model from checkpoint."""
    filename = create_filename(datadir, dataname, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename, map_location=torch.device("cpu"))
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def create_filename(save_dir, dataname, isbest=False, num_epochs=-1):
    filename = os.path.join(save_dir, dataname)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))
    return filename + ".pth.tar"


dataset = "syn1"
dataset_path = "dataset/"

A_np, X_np = load_XA(dataset, datadir=dataset_path)
num_nodes = X_np.shape[0]
labels = load_labels(dataset, datadir=dataset_path)
labels = torch.tensor(labels, dtype=torch.long).to(device)
num_class = max(labels) + 1


A = torch.tensor(A_np, dtype=torch.float32).to(device)
X = torch.tensor(X_np, dtype=torch.float32).to(device)


X = F.one_hot(torch.sum(A, 1).type(torch.LongTensor)).type(torch.float32).to(device)

input_dim = X.shape[1]

edge_index, _ = dense_to_sparse(A)
edge_index = edge_index.to(device)
num_edges = edge_index.size(1)

edge_mask = torch.ones(num_edges, dtype=torch.bool)

negative_fraction = 0.2
num_negative_samples = int(num_edges * negative_fraction)

negative_indices = random.sample(range(num_edges), num_negative_samples)
edge_mask[negative_indices] = False

negative_indices = random.sample(range(num_edges), num_negative_samples)
edge_mask[negative_indices] = False


model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=num_class).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

idx = torch.arange(len(X))
train_idx, test_idx = train_test_split(idx, train_size=0.8)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


#
#
@torch.no_grad()
def test():
    model.eval()
    pred = model(X, edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == labels[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == labels[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


pbar = tqdm(range(1, 2001))
for epoch in pbar:
    loss = train()
    if epoch == 1 or epoch % 200 == 0:
        train_acc, test_acc = test()
        pbar.set_description(
            f"Loss: {loss:.4f}, Train: {train_acc:.4f}, " f"Test: {test_acc:.4f}"
        )
pbar.close()
model.eval()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=800),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs",
    ),
)

idx = torch.arange(len(X))
targets, preds = [], []
node_indices = range(400, num_nodes, 5)
for node_index in tqdm(node_indices, leave=False, desc="Train Explainer"):
    target = labels
    explanation = explainer(X, edge_index, index=node_index, target=target)

    _, _, _, hard_edge_mask = k_hop_subgraph(
        node_index, num_hops=3, edge_index=edge_index
    )

    targets.append(edge_mask[hard_edge_mask].cpu())
    preds.append(explanation.edge_mask[hard_edge_mask].cpu())

fid_pm = fidelity(explainer, explanation)

auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
print(f"Mean ROC AUC : {auc:.4f}")
current_time1 = datetime.now()
print("Current time:", current_time1)
total_time = current_time1 - current_time
print(current_time1 - current_time)
total_milliseconds = int(total_time.total_seconds() * 1000)
print("Total time (ms):", total_milliseconds)
