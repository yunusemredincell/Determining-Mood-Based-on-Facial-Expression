import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    classification_report,
)
from torchview import draw_graph
from torchsummary import summary as model_summary
from captum.attr import GradientShap, Saliency, IntegratedGradients, LayerGradCam


class ModelEvaluation:
    def __init__(self, name, classes, device=None, root_dir="results/visualization"):
        self.classes = classes
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(root_dir, f"{name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _save_fig(self, fig, name):
        path = os.path.join(self.run_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def plot_history(self, train_losses, val_losses, train_accs, val_accs):
        epochs = range(1, len(train_losses) + 1)

        # Loss curve
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="Train Loss")
        ax.plot(epochs, val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        self._save_fig(fig, "loss_curve.png")

        # Accuracy curve
        fig, ax = plt.subplots()
        ax.plot(epochs, train_accs, label="Train Acc")
        ax.plot(epochs, val_accs, label="Val Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training & Validation Accuracy")
        ax.legend()
        self._save_fig(fig, "accuracy_curve.png")

    def plot_class_distribution(self, loader, split):
        counts = {c: 0 for c in self.classes}
        for _, targets in loader:
            for t in targets.numpy():
                counts[self.classes[t]] += 1
        fig, ax = plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
        ax.set_title(f"{split} set class distribution")
        self._save_fig(fig, f"{split}_class_distribution.png")

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        self._save_fig(fig, "confusion_matrix.png")

    def plot_roc(self, y_true, y_score, n_classes):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fig, ax = plt.subplots()
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f"{self.classes[i]} (AUC = {roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        self._save_fig(fig, "roc_curve.png")

    def _normalize_for_display(self, img_tensor):
        img = img_tensor.clone()

        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        return img

    def plot_classification_report(self, y_true, y_pred):
        report_dict = classification_report(
            y_true, y_pred, target_names=self.classes, output_dict=True
        )
        df = pd.DataFrame(report_dict).T

        display_df = df.loc[self.classes + ["macro avg", "weighted avg"]]
        metrics_df = (
            display_df[["precision", "recall", "f1-score"]].astype(float).round(2)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind="bar", ax=ax)

        ax.set_title("Classification Report")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

        self._save_fig(fig, "classification_report.png")

    def _display_original_image(self, ax, data):
        if data.shape[0] == 1:
            normalized_img = self._normalize_for_display(data.cpu().squeeze())
            ax.imshow(normalized_img, cmap="gray")
        else:
            normalized_img = self._normalize_for_display(data.cpu().permute(1, 2, 0))
            ax.imshow(normalized_img)

        ax.set_title("Original Image")
        ax.axis("off")

    def _handle_grayscale_and_rgb(self, data):
        if data.shape[1] == 1:
            return data.squeeze().cpu()
        return torch.norm(data.squeeze().cpu(), dim=0)

    def visualize_saliency_map(self, model, data, target):
        saliency = Saliency(model)
        grads = saliency.attribute(data.unsqueeze(0), target)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        self._display_original_image(ax[0], data)
        saliency_map = self._handle_grayscale_and_rgb(grads)

        ax[1].imshow(saliency_map, cmap="hot")
        ax[1].set_title("Saliency Map")
        ax[1].axis("off")
        self._save_fig(fig, "saliency_map.png")

    def visualize_integrated_gradients(self, model, data, target):
        ig = IntegratedGradients(model)
        attributions, _ = ig.attribute(
            data.unsqueeze(0), target=target, return_convergence_delta=True
        )

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        self._display_original_image(ax[0], data)
        attr_map = self._handle_grayscale_and_rgb(attributions)

        ax[1].imshow(attr_map, cmap="hot")
        ax[1].set_title("Integrated Gradients")
        ax[1].axis("off")
        self._save_fig(fig, "integrated_gradients.png")

    def visualize_gradient_shap(self, model, data, target):
        gs = GradientShap(model)
        baseline_samples = []

        for _ in range(10):
            baseline_samples.append(torch.zeros_like(data))
        baselines = torch.stack(baseline_samples)  # Shape: (10, C, H, W)

        attr = gs.attribute(data.unsqueeze(0), baselines=baselines, target=target)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        self._display_original_image(ax[0], data)
        feat_map = self._handle_grayscale_and_rgb(attr)
        ax[1].imshow(feat_map, cmap="hot")
        ax[1].set_title("Gradient SHAP")
        ax[1].axis("off")
        self._save_fig(fig, "feature_importance.png")

    def visualize_explanations(self, model, data, target):
        self.visualize_saliency_map(model, data, target)
        self.visualize_integrated_gradients(model, data, target)
        self.visualize_gradient_shap(model, data, target)

    def model_summary(self, model, data_loader, acc):
        path = os.path.join(self.run_dir, "model_summary")
        input_tensor = data_loader.dataset[0][0].unsqueeze(0)  # Add batch dimension
        input_size = tuple(input_tensor.shape)

        with open(f"{path}.txt", "w") as f:
            summary = str(model_summary(model, input_size=input_size))
            f.write(summary)
            f.write(f"\nTest Accuracy:{acc*100:.2f}%")

        model_graph = draw_graph(model, input_size=input_size, expand_nested=True)
        model_graph.visual_graph.render(f"{path}.gv", format="svg")

    @torch.no_grad
    def evaluate(self, model: nn.Module, test_loader: DataLoader, history=None):
        if history:
            self.plot_history(
                history["train_losses"],
                history["val_losses"],
                history["train_accs"],
                history["val_accs"],
            )

        model.eval()
        self.plot_class_distribution(test_loader, "test")

        all_preds = []
        all_scores = []
        all_targets = []

        for data, targets in test_loader:
            data = data.to(self.device)
            outputs = model(data)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)
        all_targets = np.array(all_targets)

        acc = accuracy_score(all_targets, all_preds)
        self.plot_classification_report(all_targets, all_preds)

        self.plot_confusion_matrix(all_targets, all_preds)
        self.plot_roc(all_targets, all_scores, len(self.classes))
        self.model_summary(model, test_loader, acc)

        data_batch, targets_batch = next(iter(test_loader))
        sample_data = data_batch[0].to(self.device)
        sample_target = targets_batch[0].item()
        self.visualize_explanations(model, sample_data, sample_target)

        print(f"Evaluation plots saved to {self.run_dir}")
