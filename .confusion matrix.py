import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Example data — replace with your actual y_true and y_pred
y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # 0=Benign, 1=Malignant
y_pred = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)
classes = ['Benign', 'Malignant']

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('fig6_confusion_matrix.png', dpi=300)
plt.show()
