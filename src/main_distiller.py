import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

from .model import (
    SGC,
    TeacherGCN,
    StudentModel,
    criterion_inter_layer,
    criterion_distill,
)
from .utils import mask, save_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(adj, x, train_mask, y, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x, adj)[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(adj, x, train_mask, y, model):
    model.eval()
    logits = model(x, adj)
    for mask in [train_mask]:
        pred = torch.argmax(logits[mask], dim=1)
        acc = pred.eq(torch.argmax(y[mask], dim=1)).sum().item() / mask.sum().item()
    return acc


def train_distill(
    edge_index,
    x,
    train_mask,
    y,
    teacher,
    student,
    optimizer,
    alpha=0.5,
    temperature=3.0,
):
    student.train()
    teacher.eval()

    optimizer.zero_grad()
    with torch.no_grad():
        teacher_features = teacher(x, edge_index, return_hidden=True)

    student_features = student(x, edge_index, return_hidden=True)

    teacher_features_train = [feat[train_mask] for feat in teacher_features]
    student_features_train = [feat[train_mask] for feat in student_features]
    y_train = y[train_mask]

    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        labels = torch.argmax(y_train, dim=1)
    else:
        labels = y_train

    total_loss, loss_kl, loss_ce = criterion_distill(
        student_features_train,
        teacher_features_train,
        labels,
        alpha=alpha,
        temperature=temperature,
    )

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_kl.item(), loss_ce.item()


def test_distill(edge_index, x, mask, y, model):
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, return_hidden=False)
        pred = torch.argmax(logits[mask], dim=1)

        if len(y.shape) > 1 and y.shape[1] > 1:
            target = torch.argmax(y[mask], dim=1)
        else:
            target = y[mask]

        acc = pred.eq(target).sum().item() / mask.sum().item()
    return acc


def main_distill(
    dataset_name,
    database,
    num_epochs,
    iterations,
    input_dim,
    hidden_dim,
    num_class,
    num_layers,
    out_path,
    teacher_path=None,
    alpha=0.5,
    temperature=3.0,
    lr=0.01,
    weight_decay=5e-4,
):
    avg_acc = []

    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")

        X, A, edge_index, L_model = database

        teacher = TeacherGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_class,
            num_layers=num_layers,
        ).to(device)

        student = StudentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_class,
            num_layers=num_layers,
            k=1,
        ).to(device)

        X = X.to(device)
        edge_index = edge_index.to(device)
        L_model = L_model.to(device)

        if teacher_path is not None:
            print(f"Loading pre-trained teacher from {teacher_path}")
            checkpoint = torch.load(teacher_path, map_location=device)
            teacher.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Training teacher model...")
            teacher = train_teacher(
                teacher,
                X,
                edge_index,
                L_model,
                num_epochs=200,
                lr=0.01,
                weight_decay=5e-4,
            )

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            student.parameters(), lr=lr, weight_decay=weight_decay
        )

        mask_train, mask_val, mask_test = mask(A)
        mask_train = mask_train.numpy()
        mask_val = mask_val.numpy()
        mask_test = mask_test.numpy()

        best_acc = 0
        best_model = None

        for epoch in tqdm(range(1, num_epochs + 1)):
            total_loss, loss_kl, loss_ce = train_distill(
                edge_index,
                X,
                mask_train,
                L_model,
                teacher,
                student,
                optimizer,
                alpha=alpha,
                temperature=temperature,
            )

            train_acc = test_distill(edge_index, X, mask_train, L_model, student)
            val_acc = test_distill(edge_index, X, mask_val, L_model, student)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = student.state_dict().copy()

            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}: Loss={total_loss:.4f} (KL={loss_kl:.4f}, CE={loss_ce:.4f}), "
                    f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
                )

        student.load_state_dict(best_model)
        test_acc = test_distill(edge_index, X, mask_test, L_model, student)
        print(f"Test Accuracy: {test_acc:.4f}")

        avg_acc.append(test_acc)
        print(
            f"Mean Accuracy: {np.mean(avg_acc):.4f} | Std Accuracy: {np.std(avg_acc):.4f}"
        )

        save_data = {
            "adj": A,
            "feat": X,
            "label": L_model,
        }

        save_checkpoint(
            best_model,
            optimizer,
            out_path,
            dataset_name + "_student",
            num_epochs=num_epochs,
            save_data=save_data,
        )

    return np.mean(avg_acc), np.std(avg_acc)


def train_teacher(
    teacher, X, edge_index, labels, num_epochs=200, lr=0.01, weight_decay=5e-4
):
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)

    num_nodes = X.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: int(num_nodes * 0.8)] = True
    train_mask = train_mask.numpy()

    if len(labels.shape) > 1 and labels.shape[1] > 1:
        target = torch.argmax(labels, dim=1)
    else:
        target = labels

    for epoch in range(num_epochs):
        teacher.train()
        optimizer.zero_grad()

        logits = teacher(X, edge_index, return_hidden=False)
        loss = F.cross_entropy(logits[train_mask], target[train_mask])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            teacher.eval()
            with torch.no_grad():
                pred = torch.argmax(logits[train_mask], dim=1)
                acc = pred.eq(target[train_mask]).sum().item() / train_mask.sum()
            print(
                f"Teacher Epoch {epoch+1}: Loss={loss.item():.4f}, Train Acc={acc:.4f}"
            )

    return teacher


def main(
    dataset_name,
    database,
    num_epochs,
    iterations,
    layer,
    input_dim,
    num_class,
    out_path,
):
    avg = []

    for i in range(iterations):
        train_acc_list = []
        best_acc = 0
        epochs_no_improve = 0

        X, A, edge_index, L_model = database

        model = SGC(layer, input_dim, num_class).to(device)
        X = X.to(device)
        edge_index = edge_index.to(device)
        L_model = L_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-6)

        criterion = F.cross_entropy

        mask_train, mask_val, mask_test = mask(A)
        for epoch in tqdm(range(1, num_epochs)):
            loss = train(
                edge_index, X, mask_train.numpy(), L_model, model, criterion, optimizer
            )
            train_acc = test(edge_index, X, mask_train.numpy(), L_model, model)
            if train_acc > best_acc:
                best_acc = train_acc
                best_model = model.state_dict()

            log = "Epoch: {:03d},loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
            train_acc_list.append(train_acc)
        avg.append(train_acc)
        print("Mean Accuracy:{} |  Std Accuracy:{}".format(np.mean(avg), np.std(avg)))

        save_data = {
            "adj": A,
            "feat": X,
            "label": L_model,
        }

        save_checkpoint(
            best_model,
            optimizer,
            out_path,
            dataset_name,
            num_epochs=epoch,
            save_data=save_data,
        )
