import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
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

# === Reshape for VGG16 input ===
def reshape_to_vgg_input(X):
    n_samples, n_features = X.shape
    target_size = 224 * 224  # 50176
    if n_features < target_size:
        X = np.hstack([X, np.zeros((n_samples, target_size - n_features))])
    elif n_features > target_size:
        X = X[:, :target_size]
    return X.reshape(-1, 224, 224, 1)

X_reshaped = reshape_to_vgg_input(X_scaled)

# Convert 1 channel to 3 for VGG16
X_vgg_input = np.repeat(X_reshaped, 3, axis=-1)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_vgg_input, y_categorical, test_size=0.3, random_state=0)

# === Build VGG16 + CNN Model ===
def build_vgg16_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze VGG16 weights

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

# === Compile & Train ===
model = build_vgg16_cnn_model(input_shape=(224, 224, 3), num_classes=y_categorical.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32)

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
--------------------------------------------------------
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# === Load CSV File with Image Paths and Labels ===
csv_path = ""
df = pd.read_csv(csv_path)

# === Remove image path column if exists ===
if 'image_path' in df.columns:
    df = df.drop(columns=['image_path'])

# === Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
y_cat = to_categorical(y)

# === Extract LBP Features ===
def extract_lbp_features(row, P=8, R=1):
    features = row.drop('label').values.astype('float32')
    size = int(np.sqrt(len(features)))
    image = features.reshape((size, size))
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2), density=True)
    return hist

X_lbp = np.vstack(df.apply(extract_lbp_features, axis=1))

# === Normalize LBP Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_lbp)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.3, random_state=0)

# === Build LBPNet (Shallow Neural Network) ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

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

metrics = calculate_metrics(y_true, y_pred)

# === Print Metrics ===
print("\n✅ Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k:20}: {v:.4f}")

---------------------------------------------------------------
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

# === Load CSV File with Features and Labels ===
csv_path = ""
df = pd.read_csv(csv_path)

# === Drop image_path column if exists ===
if 'image_path' in df.columns:
    df = df.drop(columns=['image_path'])

# === Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
y_cat = to_categorical(y)

# === Extract LBP Features ===
def extract_lbp_features(row, P=8, R=1):
    features = row.drop('label').values.astype('float32')
    size = int(np.sqrt(len(features)))
    image = features.reshape((size, size))
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2), density=True)
    return hist

X_lbp = np.vstack(df.apply(extract_lbp_features, axis=1))

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_lbp)

# === Reshape to image-like for DFDT ===
def reshape_to_image_like(X):
    n_samples, n_features = X.shape
    side = int(np.ceil(np.sqrt(n_features)))
    new_size = side * side
    if new_size != n_features:
        pad = np.zeros((n_samples, new_size - n_features))
        X = np.hstack((X, pad))
    return X.reshape(n_samples, side, side, 1), (side, side, 1)

X_reshaped, input_shape = reshape_to_image_like(X_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.3, random_state=0)

# === Patch Extractor Layer ===
class PatchExtract(layers.Layer):
    def __init__(self, patch_size=4):
        super().__init__()
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

# === DFDT Model ===
def build_dfdt_model(input_shape, patch_size=4, projection_dim=64, transformer_layers=2, num_heads=4, num_classes=2):
    inputs = Input(shape=input_shape)

    # CNN Frontend
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Vision Transformer
    patches = PatchExtract(patch_size)(x)
    x = Dense(projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization()(x)
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([x, attn])
        x3 = LayerNormalization()(x2)
        mlp = tf.keras.Sequential([
            Dense(projection_dim * 2, activation='relu'),
            Dense(projection_dim)
        ])
        x = layers.Add()([x2, mlp(x3)])

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="DFDT_Model")

# === Compile and Train ===
model = build_dfdt_model(input_shape=input_shape, num_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# === Evaluation Metrics ===
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificity_per_class, npv_per_class, fpr_per_class, fnr_per_class = [], [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

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

# === Predict and Report ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

metrics = calculate_metrics(y_true, y_pred)

print("\n✅ Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k:20}: {v:.4f}")
-----------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# === Load CSV File ===
csv_path = ""
df = pd.read_csv(csv_path)

# === Preprocess ===
X = df.drop('label', axis=1).values
y = df['label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.3, random_state=0)

# === RBM Stack (Unsupervised Pretraining) ===
# You can stack more RBMs if needed
rbm1 = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=10, random_state=0)
rbm2 = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=10, random_state=0)

# === Sequential RBM pretraining ===
X_train_rbm1 = rbm1.fit_transform(X_train)
X_train_rbm2 = rbm2.fit_transform(X_train_rbm1)

X_test_rbm1 = rbm1.transform(X_test)
X_test_rbm2 = rbm2.transform(X_test_rbm1)

# === Deep Belief Network (DBN) using Keras for fine-tuning ===
model = Sequential()
model.add(Dense(64, input_shape=(X_train_rbm2.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train DBN ===
model.fit(X_train_rbm2, y_train, validation_data=(X_test_rbm2, y_test), epochs=20, batch_size=16)

# === Evaluation ===
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificity_per_class, npv_per_class, fpr_per_class, fnr_per_class = [], [], [], []

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
y_pred_probs = model.predict(X_test_rbm2)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

results = calculate_metrics(y_true, y_pred)

# === Print Metrics ===
print("\n✅ Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20}: {v:.4f}")
----------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Load CSV File ===
csv_path = ""
df = pd.read_csv(csv_path)

# === Preprocess Data ===
X = df.drop('label', axis=1).values
y = df['label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Reshape for CNN-LSTM (samples, timesteps, features) ===
def reshape_cnn_lstm(X, max_steps=16):
    n_samples, n_features = X.shape
    steps = min(max_steps, n_features)
    f_per_step = n_features // steps
    steps = n_features // f_per_step
    X = X[:, :f_per_step * steps]
    return X.reshape(n_samples, steps, f_per_step)

X_seq = reshape_cnn_lstm(X_scaled, max_steps=16)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_cat, test_size=0.3, random_state=0)

# === Build CNN-LSTM Model ===
def build_cnn_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

model = build_cnn_lstm_model(X_train.shape[1:], num_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Metrics ===
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificity_per_class, npv_per_class, fpr_per_class, fnr_per_class = [], [], [], []

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

# === Predict and Report ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

results = calculate_metrics(y_true, y_pred)
print("\n✅ CNN-LSTM Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20}: {v:.4f}")
-------------------------------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical

# === Load and Preprocess CSV Data ===
csv_path = ""
df = pd.read_csv(csv_path)

if 'image_path' in df.columns:
    df = df.drop(columns=['image_path'])

X = df.drop('label', axis=1).values
y = df['label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Reshape into image-like format for Conv2D ===
def reshape_to_square(X):
    n_samples, n_features = X.shape
    side = int(np.ceil(np.sqrt(n_features)))
    new_features = side * side
    if new_features > n_features:
        padding = np.zeros((n_samples, new_features - n_features))
        X = np.hstack([X, padding])
    return X.reshape(n_samples, side, side, 1), (side, side, 1)

X_img, input_shape = reshape_to_square(X_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_img, y_cat, test_size=0.3, random_state=42)

# === Squash Function ===
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1.0 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

# === PrimaryCaps Layer ===
class PrimaryCaps(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=dim_capsule * n_channels,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels

    def call(self, inputs):
        output = self.conv(inputs)
        outputs = tf.reshape(output, (-1, output.shape[1] * output.shape[2] * self.n_channels, self.dim_capsule))
        return squash(outputs)

# === DigitCaps Layer ===
class DigitCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 2), -1)
        inputs_tiled = tf.tile(inputs_expanded, [1, 1, self.num_capsule, 1, 1])
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = tf.matmul(W_tiled, inputs_tiled)
        u_hat = tf.squeeze(u_hat, -1)

        b = tf.zeros_like(u_hat[..., 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)
            v = squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)

        return v

# === Build CapsNet Model ===
def build_capsnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = PrimaryCaps(dim_capsule=8, n_channels=8, kernel_size=3, strides=1, padding='same')(x)
    x = DigitCaps(num_capsule=num_classes, dim_capsule=16)(x)
    out_caps = layers.Lambda(lambda z: tf.norm(z, axis=-1))(x)
    model = models.Model(inputs=inputs, outputs=out_caps, name="CapsNet")
    return model

# === Compile and Train ===
model = build_capsnet(input_shape, num_classes=y_cat.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Predict and Evaluate ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# === Evaluation Metrics ===
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificity, npv, fpr, fnr = [], [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        npv.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        fnr.append(fn / (fn + tp) if (fn + tp) > 0 else 0)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, average='weighted'),
        "Specificity (avg)": np.mean(specificity),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "NPV (avg)": np.mean(npv),
        "FPR (avg)": np.mean(fpr),
        "FNR (avg)": np.mean(fnr)
    }

results = calculate_metrics(y_true, y_pred)

# === Print Results ===
print("\n✅ Capsule Network Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20}: {v:.4f}")