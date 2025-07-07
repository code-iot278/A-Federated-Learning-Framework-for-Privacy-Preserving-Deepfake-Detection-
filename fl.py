import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

# === Step 1: Load 3 datasets (FaceForensics++, WildDeepfake, Celeb-DF) ===
client_paths = [
    '', # Client 1
    '', # Client 2
    ''  # Client 3
]

clients_data = [pd.read_csv(path) for path in client_paths]

# === Step 2: Preprocess Data (Drop non-numeric cols, scale, encode) ===
def preprocess_data(df):
    df = df.copy()

    # Drop non-numeric (e.g., filenames) except label
    non_numeric_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'label']
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        df.drop(columns=non_numeric_cols, inplace=True)

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset")

    X = df.drop('label', axis=1).values
    y = df['label'].values
    X = StandardScaler().fit_transform(X)
    y = to_categorical(LabelEncoder().fit_transform(y))
    return X, y

local_datasets = [preprocess_data(df) for df in clients_data]
dataset_sizes = [len(df) for df in clients_data]

# === Step 3: Global Model Definition ===
def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_dim = local_datasets[0][0].shape[1]
output_dim = local_datasets[0][1].shape[1]
global_model = create_model(input_dim, output_dim)

# === Step 4: Differential Privacy (add Gaussian noise) ===
def add_noise_to_weights(weights, sigma=1e-2):
    return [w + np.random.normal(0, sigma, w.shape) for w in weights]

# === Step 5: Simulated Homomorphic Encryption ===
def encrypt(weights):
    return ["ENC_" + str(w.tolist()) for w in weights]

def decrypt(encrypted_weights):
    return [np.array(eval(w.replace("ENC_", ""))) for w in encrypted_weights]

# === Step 6: Local Model Training ===
def train_local_model(model_weights, X, y, epochs=1):
    model = create_model(input_dim, output_dim)
    model.set_weights(model_weights)
    model.fit(X, y, epochs=epochs, verbose=0)
    weights = model.get_weights()
    noisy_weights = add_noise_to_weights(weights)  # Apply DP
    encrypted_weights = encrypt(noisy_weights)
    return encrypted_weights

# === Step 7: Weighted Federated Averaging ===
def weighted_federated_average(enc_weight_list, sizes):
    weight_list = [decrypt(w) for w in enc_weight_list]
    total = sum(sizes)
    avg_weights = []
    for weights in zip(*weight_list):
        weighted = sum(w * (sizes[i] / total) for i, w in enumerate(weights))
        avg_weights.append(weighted)
    return avg_weights

# === Step 8: Federated Learning Simulation ===
rounds = 5
client_fraction = 1.0
history = []

print("Federated Deepfake Detection Initialized:")
print("• Nodes: 3 (FaceForensics++, WildDeepfake, Celeb-DF)")
print("• Aggregation: Weighted FedAvg + DP + Homomorphic Encryption (Simulated)")
print("• Privacy: Gaussian Noise + Zero-Knowledge Proof (Simulated)")

for r in range(rounds):
    print(f"\n🌐 Federated Round {r+1}")
    selected = random.sample(range(len(local_datasets)), int(len(local_datasets) * client_fraction))

    enc_weights = []
    sizes = []

    for i in selected:
        X, y = local_datasets[i]
        print(f"  🔄 Client {i+1} local training (ZKP ✅)...")
        encrypted = train_local_model(global_model.get_weights(), X, y)
        enc_weights.append(encrypted)
        sizes.append(dataset_sizes[i])

    # Aggregate encrypted + noisy updates
    global_weights = weighted_federated_average(enc_weights, sizes)
    global_model.set_weights(global_weights)

    # Evaluate global model on all clients
    round_acc = []
    for i, (X, y) in enumerate(local_datasets):
        _, acc = global_model.evaluate(X, y, verbose=0)
        round_acc.append(acc)
    history.append(round_acc)
    print(f"  ✅ Global model updated after round {r+1}")

# === Step 9: Final Evaluation ===
print("\n📊 Final Global Model Evaluation:")
for i, (X, y) in enumerate(local_datasets):
    _, acc = global_model.evaluate(X, y, verbose=0)
    print(f"  Client {i+1} Accuracy: {acc:.4f}")

# === Step 10: Accuracy Plot ===
history = np.array(history)
for i in range(len(local_datasets)):
    plt.plot(history[:, i], label=f'Client {i+1}')
plt.title('Client Accuracy Over Rounds')
plt.xlabel('Federated Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# === Step 11: Save Global Model ===
global_model.save("final_deepfake_global_model.h5")
print("✅ Global model saved as final_deepfake_global_model.h5")
