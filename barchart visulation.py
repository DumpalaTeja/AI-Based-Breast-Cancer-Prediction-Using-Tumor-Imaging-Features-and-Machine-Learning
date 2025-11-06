import matplotlib.pyplot as plt

# Metrics from your model
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [98.23, 98.43, 98.98, 98.71]

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, values, color=['#5DADE2', '#58D68D', '#F5B041', '#AF7AC5'])
plt.ylim(0, 100)
plt.ylabel('Percentage (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add labels above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('fig5_bar_metrics.png', dpi=300)
plt.show()
