import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ── Configuration ───────────────────────────────────────────────────────────────
DATA_DIR  = './data'
MODEL_DIR = './model'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load dataset ────────────────────────────────────────────────────────────────
print("=" * 55)
print("  ASL Model Trainer")
print("=" * 55)

dataset_path = os.path.join(DATA_DIR, 'dataset.pkl')

# If pickle exists, load it directly
if os.path.exists(dataset_path):
    print("  📂 Loading dataset from pickle...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    X = np.array(data['data'])
    y = np.array(data['labels'])

# Otherwise rebuild from saved .npy files
else:
    print("  📂 Rebuilding dataset from .npy files...")
    X, y = [], []
    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith('.npy'):
                landmarks = np.load(os.path.join(label_dir, fname))
                X.append(landmarks)
                y.append(label)
    X = np.array(X)
    y = np.array(y)

print(f"  ✅ Dataset loaded — {len(X)} samples across {len(set(y))} classes")
print(f"  📐 Feature shape: {X.shape}")

# ── Train / test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)
print(f"\n  🔀 Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ── Train model ─────────────────────────────────────────────────────────────────
print("\n  🏋️  Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  ✅ Training complete!")

# ── Evaluate ────────────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n  📊 Test Accuracy: {accuracy * 100:.2f}%")
print("\n  📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ── Plot accuracy per letter ────────────────────────────────────────────────────
labels     = sorted(set(y))
per_class  = []
for label in labels:
    idx       = y_test == label
    if idx.sum() > 0:
        acc   = accuracy_score(y_test[idx], y_pred[idx])
        per_class.append(acc * 100)
    else:
        per_class.append(0)

plt.figure(figsize=(14, 5))
bars = plt.bar(labels, per_class, color='steelblue', edgecolor='black')
plt.axhline(y=accuracy * 100, color='red', linestyle='--', label=f'Overall: {accuracy*100:.1f}%')
plt.title('Per-Letter Accuracy — ASL Classifier', fontsize=14)
plt.xlabel('Letter')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.legend()
for bar, val in zip(bars, per_class):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'accuracy_chart.png'))
plt.show()
print(f"  📈 Accuracy chart saved → {MODEL_DIR}/accuracy_chart.png")

# ── Save model ──────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, 'asl_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'labels': labels}, f)

print(f"\n  💾 Model saved → {model_path}")
print(f"\n  🎯 Overall accuracy: {accuracy * 100:.2f}%")

if accuracy >= 0.90:
    print("  🟢 Great accuracy! Ready to run app.py")
elif accuracy >= 0.75:
    print("  🟡 Decent accuracy. Consider collecting more samples per letter.")
else:
    print("  🔴 Low accuracy. Try collecting more varied samples and retrain.")

print("\n  Next step: run  py -3.10 app.py  to launch the live translator!")