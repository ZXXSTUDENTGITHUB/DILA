import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import os
import numpy as np

from dig.xgraph.method import SubgraphX, GradCAM, FlowX
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import find_closest_node_result
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def custom_collate(batch):
    batch = [item for item in batch if item is not None]

    return default_collate(batch)


class MyGraphDataset(Dataset):
    def __init__(self, A_list, X_list, labels_list, edge_index_list):
        self.A_list = A_list
        self.X_list = X_list
        self.labels_list = labels_list
        self.edge_index_list = edge_index_list

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        print(f"Index: {idx}, Length of edge_index_list: {len(self.edge_index_list)}")

        A = torch.tensor(self.A_list[idx], dtype=torch.float32)
        X = torch.tensor(self.X_list[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels_list[idx], dtype=torch.long)

        if idx < len(self.edge_index_list):
            edge_index = torch.tensor(self.edge_index_list[idx], dtype=torch.long)
        else:
            edge_index = None

        return {"A": A, "X": X, "labels": labels, "edge_index": edge_index}


dataset = "syn5"
dataset_path = "dataset/"

checkpoint = torch.load("trained_gcn/syn5.pth.tar")
save_data = checkpoint["save_data"]
pred = torch.tensor(save_data["pred"])

A_np, X_np = load_XA(dataset, datadir=dataset_path)
index = torch.arange(len(X_np), dtype=torch.long)
num_nodes = X_np.shape[0]
labels = load_labels(dataset, datadir=dataset_path)
labels = torch.tensor(labels, dtype=torch.long).to(device)
num_class = max(labels) + 1


A = torch.tensor(A_np, dtype=torch.float32).to(device)
X = torch.tensor(X_np, dtype=torch.float32).to(device)

if dataset == "syn1":
    one_hot_tensor = (
        F.one_hot(torch.sum(A, 1).type(torch.LongTensor)).type(torch.float32).to(device)
    )
    X = torch.cat([one_hot_tensor, X], 1)
else:
    X = F.one_hot(torch.sum(A, 1).type(torch.LongTensor)).type(torch.float32).to(device)

input_dim = X.shape[1]
edge_index, _ = dense_to_sparse(A)
edge_index = edge_index.to(device)

A_list = [torch.tensor(A_np[i], dtype=torch.float32) for i in range(len(A_np))]
X_list = [torch.tensor(X_np[i], dtype=torch.float32) for i in range(len(X_np))]
labels_list = [torch.tensor(labels[i], dtype=torch.long) for i in range(len(labels))]
edge_index_list = [
    torch.tensor(edge_index[i], dtype=torch.long) for i in range(len(edge_index))
]

my_dataset = MyGraphDataset(A_list, X_list, labels_list, edge_index_list)
batch_size = 64
dataloader = DataLoader(
    my_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
)

gnn_model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=num_class).to(device)
gnn_model = gnn_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    gnn_model.train()
    optimizer.zero_grad()
    pred = gnn_model(X, edge_index)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


explainer = FlowX(gnn_model, explain_graph=True)
sparsity = 0.5
x_collector = XCollector(sparsity)

for index, data in enumerate(dataloader):

    print(data.keys())

    if torch.isnan(data["labels"][0].squeeze()):
        continue

    walks, masks, related_preds = explainer(
        data["X"], data["edge_index"], sparsity=sparsity, num_classes=num_class
    )

    x_collector.collect_data(
        masks, related_preds, data["labels"][0].squeeze().long().item()
    )

    # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
    # obtain the result: x_processor(data, masks, x_collector)

    if index >= 99:
        break

print(x_collector.accuracy)
