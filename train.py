import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TeacherGCN, StudentModel, criterion_distill
from src.evaluation import (
    DistillationTimer,
    compute_distillation_accuracy,
    compute_model_accuracy,
    compute_explanation_accuracy,
    generate_explanations,
    evaluate_all_metrics,
)
from src.utils import (
    load_XA,
    load_labels,
    mask,
    get_nodes_explained,
    evaluate_syn_explanation,
    get_ground_truth,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def prepare_data(dataset_name, dataset_path="dataset/"):
    A, X = load_XA(dataset_name, datadir=dataset_path)
    labels = load_labels(dataset_name, datadir=dataset_path)

    A = torch.tensor(A, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    if dataset_name == "syn1":
        one_hot = F.one_hot(torch.sum(A, 1).long()).float()
        X = torch.cat([one_hot, X], dim=1)
    else:
        X = F.one_hot(torch.sum(A, 1).long()).float()

    edge_index, _ = dense_to_sparse(A)

    num_classes = int(labels.max().item() + 1)
    labels_onehot = F.one_hot(labels, num_classes).float()

    return X, A, edge_index, labels, labels_onehot, num_classes


def train_teacher_model(
    teacher,
    X,
    edge_index,
    labels,
    mask_train,
    num_epochs=200,
    lr=0.01,
    weight_decay=5e-4,
):
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        teacher.train()
        optimizer.zero_grad()

        logits = teacher(X, edge_index, return_hidden=False)
        loss = F.cross_entropy(logits[mask_train], labels[mask_train])

        loss.backward()
        optimizer.step()

        teacher.eval()
        with torch.no_grad():
            pred = torch.argmax(logits[mask_train], dim=1)
            acc = (pred == labels[mask_train]).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = teacher.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"  Teacher Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    teacher.load_state_dict(best_state)
    return teacher


def train_student_with_distillation(
    teacher,
    student,
    X,
    edge_index,
    labels,
    mask_train,
    mask_val,
    num_epochs=200,
    alpha=0.5,
    temperature=3.0,
    lr=0.01,
    weight_decay=5e-4,
):

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_state = None

    for epoch in range(num_epochs):
        student.train()
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_features = teacher(X, edge_index, return_hidden=True)

        student_features = student(X, edge_index, return_hidden=True)

        teacher_feat_train = [f[mask_train] for f in teacher_features]
        student_feat_train = [f[mask_train] for f in student_features]

        total_loss, loss_kl, loss_ce, layer_losses = criterion_distill(
            student_feat_train,
            teacher_feat_train,
            labels[mask_train],
            alpha=alpha,
            temperature=temperature,
            align_mode="combined",
        )

        total_loss.backward()
        optimizer.step()

        student.eval()
        with torch.no_grad():
            val_logits = student(X, edge_index, return_hidden=False)
            val_pred = torch.argmax(val_logits[mask_val], dim=1)
            val_acc = (val_pred == labels[mask_val]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = student.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            train_logits = student(X, edge_index, return_hidden=False)
            train_pred = torch.argmax(train_logits[mask_train], dim=1)
            train_acc = (train_pred == labels[mask_train]).float().mean().item()
            layer_loss_str = ", ".join(
                [f"L{i+1}={l:.4f}" for i, l in enumerate(layer_losses)]
            )
            print(
                f"  Student Epoch {epoch+1}: Loss={total_loss.item():.4f} "
                f"(KL={loss_kl.item():.4f}, CE={loss_ce.item():.4f}), "
                f"Layers=[{layer_loss_str}], "
                f"Train={train_acc:.4f}, Val={val_acc:.4f}"
            )

    student.load_state_dict(best_state)
    return student


def run_experiment(
    dataset_name,
    num_epochs=200,
    hidden_dim=64,
    num_layers=3,
    alpha=0.5,
    temperature=3.0,
    iterations=3,
):

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    X, A, edge_index, labels, labels_onehot, num_classes = prepare_data(dataset_name)
    input_dim = X.shape[1]

    print(f"Nodes: {X.shape[0]}, Features: {input_dim}, Classes: {num_classes}")

    node_list, k = get_nodes_explained(dataset_name, A.numpy())
    print(f"Nodes to explain: {len(node_list)}, k-hop: {k}")

    all_metrics = {
        "teacher_accuracy": [],
        "student_accuracy": [],
        "distillation_accuracy": [],
        "distillation_time": [],
        "explanation_time": [],
        "overall_runtime": [],
        "explanation_accuracy": [],
        "explanation_precision": [],
    }

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

        mask_train, mask_val, mask_test = mask(A.numpy())
        mask_train = mask_train.to(device)
        mask_val = mask_val.to(device)
        mask_test = mask_test.to(device)

        X_dev = X.to(device)
        edge_index_dev = edge_index.to(device)
        labels_dev = labels.to(device)

        teacher = TeacherGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
        ).to(device)

        student = StudentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            k=1,
        ).to(device)

        print("\n[1] Training Teacher Model...")
        teacher = train_teacher_model(
            teacher,
            X_dev,
            edge_index_dev,
            labels_dev,
            mask_train,
            num_epochs=num_epochs,
            lr=0.01,
        )

        teacher_acc = compute_model_accuracy(
            teacher, X_dev, edge_index_dev, labels_dev, mask_test
        )
        print(f"Teacher Test Accuracy: {teacher_acc:.4f}")
        all_metrics["teacher_accuracy"].append(teacher_acc)

        print("\n[2] Distilling Student Model...")
        timer = DistillationTimer()
        timer.start()

        student = train_student_with_distillation(
            teacher,
            student,
            X_dev,
            edge_index_dev,
            labels_dev,
            mask_train,
            mask_val,
            num_epochs=num_epochs,
            alpha=alpha,
            temperature=temperature,
        )

        distill_time = timer.stop()
        print(f"Distillation Time: {distill_time:.2f}s")
        all_metrics["distillation_time"].append(distill_time)

        student_acc = compute_model_accuracy(
            student, X_dev, edge_index_dev, labels_dev, mask_test
        )
        print(f"Student Test Accuracy: {student_acc:.4f}")
        all_metrics["student_accuracy"].append(student_acc)

        distill_acc = compute_distillation_accuracy(
            student, teacher, X_dev, edge_index_dev, mask_test
        )
        print(f"Distillation Fidelity: {distill_acc:.4f}")
        all_metrics["distillation_accuracy"].append(distill_acc)

        print("\n[3] Generating Explanations...")
        exp_start = time.time()

        sample_nodes = node_list[: min(100, len(node_list))]
        explanations, _ = generate_explanations(
            student, X_dev, edge_index_dev, sample_nodes, k=k
        )

        exp_time = time.time() - exp_start
        print(f"Explanation Time: {exp_time:.2f}s")
        all_metrics["explanation_time"].append(exp_time)
        all_metrics["overall_runtime"].append(distill_time + exp_time)

        exp_acc, exp_prec = evaluate_syn_explanation(explanations, dataset_name)
        print(f"Explanation Accuracy: {exp_acc:.4f}, Precision: {exp_prec:.4f}")
        all_metrics["explanation_accuracy"].append(exp_acc)
        all_metrics["explanation_precision"].append(exp_prec)

    print(f"\n{'='*60}")
    print(f"Results Summary for {dataset_name}")
    print(f"{'='*60}")

    results = {}
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    return results


def main():

    datasets = ["syn1", "syn2", "syn3", "syn4", "syn5", "syn6"]
    num_epochs = 200
    hidden_dim = 64
    num_layers = 1
    alpha = 0.5
    temperature = 3.0
    iterations = 3

    all_results = {}

    for dataset in datasets:
        try:
            results = run_experiment(
                dataset_name=dataset,
                num_epochs=num_epochs,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                alpha=alpha,
                temperature=temperature,
                iterations=iterations,
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"Error on dataset {dataset}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Dataset':<10} {'Teacher':<12} {'Student':<12} {'Fidelity':<12} "
        f"{'Exp Acc':<12} {'Distill Time':<12}"
    )
    print("-" * 70)

    for dataset, results in all_results.items():
        teacher = results.get("teacher_accuracy", {}).get("mean", 0)
        student = results.get("student_accuracy", {}).get("mean", 0)
        fidelity = results.get("distillation_accuracy", {}).get("mean", 0)
        exp_acc = results.get("explanation_accuracy", {}).get("mean", 0)
        distill_time = results.get("distillation_time", {}).get("mean", 0)

        print(
            f"{dataset:<10} {teacher:<12.4f} {student:<12.4f} {fidelity:<12.4f} "
            f"{exp_acc:<12.4f} {distill_time:<12.2f}s"
        )

    import json

    output_file = "outputs/experiment_results.json"
    os.makedirs("outputs", exist_ok=True)

    serializable_results = {}
    for dataset, metrics in all_results.items():
        serializable_results[dataset] = {}
        for metric_name, values in metrics.items():
            serializable_results[dataset][metric_name] = {
                "mean": float(values["mean"]),
                "std": float(values["std"]),
            }

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
