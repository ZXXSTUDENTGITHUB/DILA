import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, GATConv
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class base(GATConv):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(base, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim)

        self.conv2 = GATConv(hidden_dim, hidden_dim)

        self.conv3 = GATConv(hidden_dim, int(output_dim))

    def forward(self, x, edge_index, train_idx=None):
        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))

        x = self.conv3(x, edge_index)

        if train_idx is not None:
            x_train = x[train_idx]
            return F.log_softmax(x_train, dim=1)
        else:
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

if dataset == "syn3":
    one_hot_tensor = (
        F.one_hot(torch.sum(A, 1).type(torch.LongTensor)).type(torch.float32).to(device)
    )
    X = torch.cat([one_hot_tensor, X], 1)
else:
    X = F.one_hot(torch.sum(A, 1).type(torch.LongTensor)).type(torch.float32).to(device)

input_dim = X.shape[1]
edge_index, _ = dense_to_sparse(A)
edge_index = edge_index.to(device)

gnn_model = base(input_dim=input_dim, hidden_dim=64, output_dim=num_class).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    gnn_model.train()
    optimizer.zero_grad()
    pred = gnn_model(X, edge_index, train_idx)
    loss = criterion(pred, labels[train_idx])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

save_path = "trained_gat"
save_filename = "syn1.pth.tar"
save_dict = {
    "epoch": 100,
    "model_type": base,
    "optimizer": optimizer,
    "model_state": gnn_model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "save_data": {
        "adj": A_np,
        "feat": X_np,
        "label": labels.cpu().numpy(),
        "pred": pred.detach().cpu().numpy(),
        "train_idx": train_idx,
    },
}

torch.save(save_dict, os.path.join(save_path, save_filename))
