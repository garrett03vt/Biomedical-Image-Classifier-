# AUC and accuracy calculation scripts
from medmnist import Evaluator
import numpy as np

# Get predicted probabilities (needed for AUC)
y_score = clf.predict_proba(X_test_scaled)   # shape (N, n_classes)
y_pred  = clf.predict(X_test_scaled)          # shape (N,)

# Reshape to match MedMNIST evaluator format
y_true_eval  = y_test.reshape(-1, 1)
y_score_eval = y_score  # already (N, n_classes)

evaluator = Evaluator(DATA_FLAG, "test")
metrics = evaluator.evaluate(y_score_eval)

print(f"AUC:      {metrics[1]:.4f}")
print(f"Accuracy: {metrics[2]:.4f}")