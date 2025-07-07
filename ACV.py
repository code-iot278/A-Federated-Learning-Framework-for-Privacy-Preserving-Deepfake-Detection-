import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Concatenate, Dense
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# === Load CSV File ===
csv_path = ""
df = pd.read_csv(csv_path)

# === Prepare Features and Labels ===
X = df.drop('label', axis=1).values
y = df['label'].values

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Feature Normalization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Reshape to image-like format ===
def reshape_to_image_like(X):
    n_samples, n_features = X.shape
    side = int(np.ceil(np.sqrt(n_features)))
    new_shape = side * side
    if new_shape != n_features:
        padding = np.zeros((n_samples, new_shape - n_features))
        X = np.hstack((X, padding))
    return X.reshape(n_samples, side, side, 1), (side, side, 1)

X_reshaped, input_shape = reshape_to_image_like(X_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=0)

# === Capsule Network Block ===
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

def capsule_network_block(inputs):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Reshape((-1, 8))(x)
    x = layers.Lambda(squash)(x)
    capsule_output = layers.Dense(16, activation='sigmoid')(x)
    return layers.GlobalAveragePooling1D()(capsule_output)

# === Vision Transformer Block ===
class PatchExtract(layers.Layer):
    def __init__(self, patch_size=4):
        super(PatchExtract, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

def vision_transformer_block(inputs, patch_size=4, projection_dim=64, transformer_layers=1):
    patches = PatchExtract(patch_size)(inputs)
    embedded_patches = Dense(projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization()(embedded_patches)
        attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attn_output, embedded_patches])
        x3 = layers.LayerNormalization()(x2)
        mlp = tf.keras.Sequential([
            Dense(projection_dim * 2, activation='relu'),
            Dense(projection_dim),
        ])(x3)
        embedded_patches = layers.Add()([mlp, x2])

    return layers.GlobalAveragePooling1D()(embedded_patches)

# === Adversarial Robustness Block ===
def adversarial_robustness_block(inputs):
    x = layers.GaussianNoise(0.1)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    return layers.GlobalAveragePooling2D()(x)

# === Hybrid Model ===
def build_hybrid_model(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    capsule_output = capsule_network_block(inputs)
    vit_output = vision_transformer_block(inputs)
    robust_output = adversarial_robustness_block(inputs)

    fused = Concatenate()([capsule_output, vit_output, robust_output])
    x = Dense(128, activation='relu')(fused)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="CapsViT_RobustNet")

# === Compile and Train ===
model = build_hybrid_model(input_shape=input_shape, num_classes=y_categorical.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# === Evaluation Metrics ===
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificity_per_class = []
    npv_per_class = []
    fpr_per_class = []
    fnr_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        specificity_per_class.append(specificity)
        npv_per_class.append(npv)
        fpr_per_class.append(fpr)
        fnr_per_class.append(fnr)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity (avg)": np.mean(specificity_per_class),
        "F1 Score": f1,
        "MCC": mcc,
        "NPV (avg)": np.mean(npv_per_class),
        "FPR (avg)": np.mean(fpr_per_class),
        "FNR (avg)": np.mean(fnr_per_class)
    }

# === Predict and Evaluate ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

results = calculate_metrics(y_true, y_pred)

# === Print Metrics ===
print("\n✅ Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20}: {v:.4f}")