import torch
from classifier import FacialExpressionClassifier
from models.transfer_learning import (
    FacialExpressionResnetNN,
    FacialExpressionEfficientNN,
)
from models.resemotenet import ResEmoteNet
from models.first_model import FacialExpressionNN

from visualization import ModelEvaluation


def main():
    model = FacialExpressionEfficientNN()
    classifier = FacialExpressionClassifier(
        "data", model, batch_size=64, learning_rate=1e-3
    )

    train_loader, validation_loader, test_loader = classifier.fit(
        epochs=80, unfreeze_epoch=20, transfer_learning=True, patience=15
    )
    history = {
        "train_losses": classifier.train_losses,
        "val_losses": classifier.val_losses,
        "train_accs": classifier.train_accuracies,
        "val_accs": classifier.val_accuracies,
    }
    evaluator = ModelEvaluation(name="Efficient-b1", classes=classifier.class_names)
    evaluator.plot_class_distribution(train_loader, "train")
    evaluator.plot_class_distribution(validation_loader, "validation")
    evaluator.evaluate(classifier.model, test_loader, history=history)

    classifier.model.eval()
    scripted_model = torch.jit.script(classifier.model)
    scripted_model.save("results/models/lightweight_model.pt")

if __name__ == "__main__":
    main()
