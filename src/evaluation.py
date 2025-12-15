import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.utils import k_hop_subgraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_explanation_accuracy(explanations, dataset, get_ground_truth_fn):
    gt_positive = 0
    true_positive = 0
    pred_positive = 0

    for node in explanations:
        ground_truth = get_ground_truth_fn(node, dataset)
        gt_positive += len(ground_truth)
        pred_positive += len(explanations[node])

        for ex_node in explanations[node]:
            if ex_node in ground_truth:
                true_positive += 1

    accuracy = true_positive / gt_positive if gt_positive > 0 else 0
    precision = true_positive / pred_positive if pred_positive > 0 else 0
    recall = true_positive / gt_positive if gt_positive > 0 else 0

    return accuracy, precision, recall


class DistillationTimer:

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.distillation_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.distillation_time = self.end_time - self.start_time
        return self.distillation_time

    def get_time(self):
        return self.distillation_time


def compute_distillation_accuracy(student, teacher, X, edge_index, mask):
    student.eval()
    teacher.eval()

    with torch.no_grad():
        teacher_logits = teacher(X, edge_index, return_hidden=False)
        student_logits = student(X, edge_index, return_hidden=False)

        teacher_pred = torch.argmax(teacher_logits[mask], dim=1)
        student_pred = torch.argmax(student_logits[mask], dim=1)

        fidelity = (student_pred == teacher_pred).float().mean().item()

    return fidelity


def compute_model_accuracy(model, X, edge_index, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index, return_hidden=False)
        pred = torch.argmax(logits[mask], dim=1)

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            target = torch.argmax(labels[mask], dim=1)
        else:
            target = labels[mask]

        acc = (pred == target).float().mean().item()
    return acc


def compute_explanation_auc(node_importance_scores, ground_truth_nodes, all_nodes):
    y_true = []
    y_score = []

    for node in all_nodes:
        y_true.append(1 if node in ground_truth_nodes else 0)
        y_score.append(node_importance_scores.get(node, 0))

    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return 0.5

    auc = roc_auc_score(y_true, y_score)
    return auc


def compute_robust_fidelity(
    model, X, edge_index, explanation_mask, alpha=0.5, num_samples=10
):
    model.eval()

    with torch.no_grad():
        original_logits = model(X, edge_index, return_hidden=False)
        original_pred = F.softmax(original_logits, dim=-1)

        r_fid_plus_list = []
        r_fid_minus_list = []

        for _ in range(num_samples):
            random_mask_exp = torch.bernoulli(
                torch.full_like(explanation_mask.float(), 1 - alpha)
            )
            masked_X_exp = X * (
                1 - explanation_mask.unsqueeze(-1) * (1 - random_mask_exp.unsqueeze(-1))
            )

            random_mask_non = torch.bernoulli(
                torch.full_like(explanation_mask.float(), 1 - alpha)
            )
            non_exp_mask = 1 - explanation_mask
            masked_X_non = X * (
                1 - non_exp_mask.unsqueeze(-1) * (1 - random_mask_non.unsqueeze(-1))
            )

            perturbed_logits_exp = model(masked_X_exp, edge_index, return_hidden=False)
            perturbed_pred_exp = F.softmax(perturbed_logits_exp, dim=-1)

            perturbed_logits_non = model(masked_X_non, edge_index, return_hidden=False)
            perturbed_pred_non = F.softmax(perturbed_logits_non, dim=-1)

            fid_plus = F.kl_div(
                perturbed_pred_exp.log(), original_pred, reduction="batchmean"
            ).item()
            fid_minus = F.kl_div(
                perturbed_pred_non.log(), original_pred, reduction="batchmean"
            ).item()

            r_fid_plus_list.append(fid_plus)
            r_fid_minus_list.append(fid_minus)

        r_fidelity_plus = np.mean(r_fid_plus_list)
        r_fidelity_minus = np.mean(r_fid_minus_list)

    return r_fidelity_plus, r_fidelity_minus


def compute_finetuned_fidelity(
    model,
    X,
    edge_index,
    labels,
    explanation_mask,
    beta=0.3,
    num_finetune_epochs=50,
    lr=0.001,
):
    import copy

    finetuned_model = copy.deepcopy(model)
    finetuned_model.train()

    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=lr)

    for _ in range(num_finetune_epochs):
        optimizer.zero_grad()

        random_mask = torch.bernoulli(
            torch.full((X.size(0),), 1 - beta, device=X.device)
        )
        masked_X = X * random_mask.unsqueeze(-1)

        logits = finetuned_model(masked_X, edge_index, return_hidden=False)

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            target = torch.argmax(labels, dim=1)
        else:
            target = labels

        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

    finetuned_model.eval()

    with torch.no_grad():
        original_logits = model(X, edge_index, return_hidden=False)
        original_pred_class = torch.argmax(original_logits, dim=1)

        exp_mask_X = X * explanation_mask.unsqueeze(-1)
        finetuned_logits_plus = finetuned_model(
            exp_mask_X, edge_index, return_hidden=False
        )
        finetuned_pred_plus = torch.argmax(finetuned_logits_plus, dim=1)

        non_exp_mask_X = X * (1 - explanation_mask.unsqueeze(-1))
        finetuned_logits_minus = finetuned_model(
            non_exp_mask_X, edge_index, return_hidden=False
        )
        finetuned_pred_minus = torch.argmax(finetuned_logits_minus, dim=1)

        f_fidelity_plus = (
            (finetuned_pred_plus == original_pred_class).float().mean().item()
        )
        f_fidelity_minus = (
            (finetuned_pred_minus == original_pred_class).float().mean().item()
        )

    return f_fidelity_plus, f_fidelity_minus


def evaluate_all_metrics(
    teacher,
    student,
    X,
    edge_index,
    labels,
    mask_test,
    explanations,
    dataset,
    get_ground_truth_fn,
    distillation_time,
    explanation_time=0,
):
    metrics = {}

    metrics["distillation_accuracy"] = compute_distillation_accuracy(
        student, teacher, X, edge_index, mask_test
    )

    metrics["student_accuracy"] = compute_model_accuracy(
        student, X, edge_index, labels, mask_test
    )

    metrics["teacher_accuracy"] = compute_model_accuracy(
        teacher, X, edge_index, labels, mask_test
    )

    metrics["distillation_time"] = distillation_time

    metrics["overall_runtime"] = distillation_time + explanation_time

    if explanations:
        exp_acc, exp_prec, exp_recall = compute_explanation_accuracy(
            explanations, dataset, get_ground_truth_fn
        )
        metrics["explanation_accuracy"] = exp_acc
        metrics["explanation_precision"] = exp_prec
        metrics["explanation_recall"] = exp_recall

    return metrics


class LinearExplainer:

    def __init__(self, student_model):
        self.student = student_model

    def explain_node(self, node_idx, X, edge_index, k=5):
        self.student.eval()

        neighbors, sub_edge_index, node_idx_new, edge_mask = k_hop_subgraph(
            int(node_idx), 3, edge_index, relabel_nodes=True
        )

        sub_X = X[neighbors]

        with torch.no_grad():
            original_logits = self.student(X, edge_index, return_hidden=False)
            original_pred = original_logits[node_idx]

            importance_scores = {}

            for i, neighbor in enumerate(neighbors.tolist()):
                masked_X = X.clone()
                masked_X[neighbor] = 0

                self.student.reset_cache()
                masked_logits = self.student(masked_X, edge_index, return_hidden=False)
                masked_pred = masked_logits[node_idx]

                importance = torch.norm(original_pred - masked_pred).item()
                importance_scores[neighbor] = importance

            self.student.reset_cache()
            self.student(X, edge_index)

        sorted_nodes = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        k = min(k, len(sorted_nodes))
        important_nodes = [node for node, _ in sorted_nodes[:k]]

        return important_nodes, importance_scores


def generate_explanations(student, X, edge_index, node_list, k=5):
    explainer = LinearExplainer(student)
    explanations = {}

    start_time = time.time()

    for node in node_list:
        important_nodes, _ = explainer.explain_node(node, X, edge_index, k)
        explanations[node] = important_nodes

    explanation_time = time.time() - start_time

    return explanations, explanation_time
