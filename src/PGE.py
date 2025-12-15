import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import os
import numpy as np
from torch.nn import BatchNorm1d
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer
from torch_geometric.explain.metric import fidelity
from torch_geometric.explain.metric import fidelity_curve_auc
from torch_geometric.explain.metric import groundtruth_metrics

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

explainer = Explainer(
    model=gnn_model,
    algorithm=PGExplainer(epochs=30, lr=0.003).to(device),
    explanation_type="phenomenon",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs",
    ),
)

target = labels
# Train against a variety of node-level or graph-level predictions:
for epoch in range(30):
    for i in range(len(index)):
        current_index = index[i].item()  # Indices to train against.
        loss = explainer.algorithm.train(
            epoch, gnn_model, X, edge_index, target=target, index=current_index
        )
        print(f"epoch:{epoch}, index:{current_index}, loss:{loss} ")
# for i in range(len(index)):
#     explanation = explainer(X, edge_index, target=target, index=i)
#     fid_pm = fidelity(explainer, explanation)
#     print(f'index:{i}, fid_pm:{fid_pm}')
# Get the final explanations:
# pred_prob = explainer.get_masked_prediction(X, edge_index)
# pred_mask = torch.max(pred_prob, dim=1)[0]
#
# target_mask = labels
# target_mask = target_mask.to('cuda:0')
#
# accuracy = groundtruth_metrics(pred_mask, target_mask, metrics=["accuracy"])
#
# print(accuracy)

prediction = explainer.get_masked_prediction(X, edge_index)
target_mask = explainer.get_target(prediction)

target_mask_cpu = target_mask.cpu().numpy()
labels_cpu = labels.cpu().numpy()

sklearn_accuracy = accuracy_score(target_mask_cpu, labels_cpu)
print(f"Accuracy (sklearn): {sklearn_accuracy:.4f}")
