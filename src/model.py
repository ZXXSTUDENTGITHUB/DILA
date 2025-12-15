import torch
from torch_geometric.nn import SGConv, SSGConv, GCNConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor


class TeacherGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(TeacherGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index, return_hidden=False):
        hidden_features = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_features.append(x)

        if return_hidden:
            return hidden_features
        else:
            return x


class StudentModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, k=1):
        super(StudentModel, self).__init__()
        self.num_layers = num_layers
        self.k = k

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

        self._cached_adj_norm = None

    def _compute_adj_norm(self, edge_index, num_nodes):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return SparseTensor(
            row=row, col=col, value=norm, sparse_sizes=(num_nodes, num_nodes)
        )

    def _propagate(self, x, adj_norm, k=1):
        for _ in range(k):
            x = adj_norm @ x
        return x

    def forward(self, x, edge_index, return_hidden=False):
        num_nodes = x.size(0)

        if self._cached_adj_norm is None:
            self._cached_adj_norm = self._compute_adj_norm(edge_index, num_nodes)
        adj_norm = self._cached_adj_norm

        hidden_features = []

        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = self._propagate(x, adj_norm, self.k)
            hidden_features.append(x)

        if return_hidden:
            return hidden_features
        else:
            return x

    def reset_cache(self):
        self._cached_adj_norm = None


def criterion_inter_layer(
    student_features,
    teacher_features,
    temperature=1.0,
    align_mode="mse",
    layer_weights=None,
):
    assert len(student_features) == len(
        teacher_features
    ), f"层数不匹配: student={len(student_features)}, teacher={len(teacher_features)}"

    num_layers = len(student_features)

    if layer_weights is None:
        layer_weights = [1.0 / num_layers for _ in range(num_layers)]

    total_loss = 0.0
    layer_losses = []

    for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
        is_last_layer = i == num_layers - 1

        if align_mode == "kl":
            s_prob = F.log_softmax(s_feat / temperature, dim=-1)
            t_prob = F.softmax(t_feat / temperature, dim=-1)
            layer_loss = F.kl_div(s_prob, t_prob, reduction="batchmean")
            layer_loss = layer_loss * (temperature**2)

        elif align_mode == "mse":
            layer_loss = F.mse_loss(s_feat, t_feat)

        elif align_mode == "cosine":
            cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1)
            layer_loss = (1 - cos_sim).mean()

        elif align_mode == "combined":
            if is_last_layer:
                s_prob = F.log_softmax(s_feat / temperature, dim=-1)
                t_prob = F.softmax(t_feat / temperature, dim=-1)
                layer_loss = F.kl_div(s_prob, t_prob, reduction="batchmean")
                layer_loss = layer_loss * (temperature**2)
            else:
                s_norm = F.normalize(s_feat, p=2, dim=-1)
                t_norm = F.normalize(t_feat, p=2, dim=-1)
                layer_loss = F.mse_loss(s_norm, t_norm) * 10.0

        else:
            raise ValueError(f"未知的对齐模式: {align_mode}")

        weighted_loss = layer_weights[i] * layer_loss
        total_loss += weighted_loss
        layer_losses.append(layer_loss.item())

    return total_loss, layer_losses


def criterion_distill(
    student_features,
    teacher_features,
    labels,
    alpha=0.5,
    temperature=1.0,
    align_mode="combined",
    layer_weights=None,
):
    loss_kl, layer_losses = criterion_inter_layer(
        student_features,
        teacher_features,
        temperature,
        align_mode=align_mode,
        layer_weights=layer_weights,
    )

    student_logits = student_features[-1]
    loss_ce = F.cross_entropy(student_logits, labels)

    total_loss = alpha * loss_kl + (1 - alpha) * loss_ce

    return total_loss, loss_kl, loss_ce, layer_losses


class SGC(nn.Module):

    def __init__(self, k, nfeat, nclass):
        super(SGC, self).__init__()
        self.conv1 = SSGConv(nfeat, nclass, k)

    def forward(self, x, edge_index, return_hidden=False):
        x = self.conv1(x, edge_index)
        if return_hidden:
            return [x]
        return x


class SSGC(torch.nn.Module):

    def __init__(self, k, nfeat, nclass):
        super(SSGC, self).__init__()
        self.conv1 = SSGConv(nfeat, nclass, k)

    def forward(self, x, edge_index, return_hidden=False):
        x = self.conv1(x, edge_index)
        if return_hidden:
            return [x]
        return x
