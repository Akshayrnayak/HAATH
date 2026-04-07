"""
╔══════════════════════════════════════════════════════════════╗
║  HAATH — LSTM DYNAMIC GESTURE MODEL TRAINER                  ║
║  Trains on 30-frame sequences using TensorFlow/Keras LSTM    ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:
    python ml_model/train_lstm.py

WHAT IT DOES:
    Loads all .npy sequence files from dataset/sequences/
    Builds a 2-layer LSTM model
    Trains with early stopping
    Saves: isl_lstm_model.h5  label_encoder.pkl  training_stats.json
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score

# ─── CONFIG ────────────────────────────────────────────────
DATASET_DIR     = "dataset/sequences"
MODEL_DIR       = "ml_model"
MODEL_PATH      = os.path.join(MODEL_DIR, "isl_lstm_model.h5")
ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
STATS_PATH      = os.path.join(MODEL_DIR, "training_stats.json")

SEQUENCE_LENGTH = 30    # frames per sequence
N_FEATURES      = 63    # 21 landmarks × 3 (x,y,z)
BATCH_SIZE      = 32
EPOCHS          = 200   # early stopping will cut this short
LEARNING_RATE   = 0.001
TEST_SIZE       = 0.2
# ───────────────────────────────────────────────────────────

def load_dataset():
    print("\n" + "="*60)
    print("  LOADING DATASET")
    print("="*60)

    X, y = [], []
    gestures = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not gestures:
        raise FileNotFoundError(
            f"No gesture folders found in {DATASET_DIR}. "
            "Run collect_dynamic.py first."
        )

    for gesture in gestures:
        folder = os.path.join(DATASET_DIR, gesture)
        files  = [f for f in os.listdir(folder) if f.endswith('.npy')]
        for fname in files:
            seq = np.load(os.path.join(folder, fname))  # (30, 63)
            if seq.shape == (SEQUENCE_LENGTH, N_FEATURES):
                X.append(seq)
                y.append(gesture)

    X = np.array(X, dtype=np.float32)  # (N, 30, 63)
    y = np.array(y)

    print(f"  Total sequences : {len(X)}")
    print(f"  Gestures found  : {len(gestures)}")
    print(f"  Input shape     : {X.shape}")
    print(f"\n  Distribution:")
    for g in gestures:
        count = np.sum(y == g)
        bar   = "█" * (count // 5)
        print(f"    {g:15} {count:4}  {bar}")

    return X, y, gestures

def augment_sequences(X, y, factor=6):
    """
    Generate 6x more data from existing sequences using
    multiple augmentation techniques — simulates recording
    variation without actually re-recording.
    """
    X_aug, y_aug = [X], [y]

    for i in range(factor - 1):
        aug = X.copy()

        # 1. Random Gaussian noise (simulates slight hand tremor)
        noise = np.random.normal(0, 0.006 + i * 0.002, aug.shape)
        aug = aug + noise.astype(np.float32)

        # 2. Random time shift (simulates starting gesture earlier/later)
        shift = np.random.randint(1, 5)
        aug = np.roll(aug, shift, axis=1)

        # 3. Random speed scaling (simulates fast vs slow performance)
        if i % 2 == 0:
            indices = np.linspace(0, aug.shape[1]-1, aug.shape[1])
            scale   = np.random.uniform(0.85, 1.15)
            new_idx = np.clip(indices * scale, 0, aug.shape[1]-1).astype(int)
            aug     = aug[:, new_idx, :]

        # 4. Random spatial scaling (simulates different distances)
        scale_factor = np.random.uniform(0.88, 1.12)
        aug = aug * scale_factor

        # 5. Random finger axis flip (simulates left vs right hand)
        if i % 3 == 0:
            aug[:, :, 0] = -aug[:, :, 0]  # flip x axis

        X_aug.append(aug)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)



def build_lstm_model(n_classes, sequence_len, n_features):
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.001),    # penalizes memorization
                 recurrent_regularizer=l2(0.001)),
            input_shape=(sequence_len, n_features)
        ),
        BatchNormalization(),
        Dropout(0.5),

        LSTM(128, return_sequences=False,
             kernel_regularizer=l2(0.001),
             recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name="ISL_Haath_LSTM_v2")

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────
    X, y_raw, gestures = load_dataset()

    # ── Encode labels ─────────────────────────────────────
    le = LabelEncoder()
    le.fit(gestures)
    y_enc = le.transform(y_raw)
    y_cat = to_categorical(y_enc, num_classes=len(gestures))

    # ── Augment ───────────────────────────────────────────
    print("\n  Augmenting sequences...")
    X_aug, y_enc_aug = augment_sequences(X, y_enc, factor=2)
    y_cat_aug = to_categorical(y_enc_aug, num_classes=len(gestures))
    print(f"  Original: {len(X)} → Augmented: {len(X_aug)} sequences")

    # ── Train/test split ───────────────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=99)
    f   or train_idx, test_idx in sss.split(X_aug, y_enc_aug):
    X_tr = X_aug[train_idx]
    X_te = X_aug[test_idx]
    y_tr = y_cat_aug[train_idx]
    y_te = y_cat_aug[test_idx]
    
    print(f"\n  Train: {len(X_tr)} | Test: {len(X_te)}")

    # ── Build model ───────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURE")
    print("="*60)
    model = build_lstm_model(len(gestures), SEQUENCE_LENGTH, N_FEATURES)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
    ]

    # ── Train ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING — (early stopping will stop when accuracy plateaus)")
    print("="*60)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATION ON TEST SET")
    print("="*60)
    y_pred_proba = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_te, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print(f"\n  Per-gesture Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=3,
        labels=range(len(le.classes_))
    )
    for line in report.split('\n'):
        print(f"    {line}")

    # ── Save artifacts ────────────────────────────────────
    # Model already saved by ModelCheckpoint
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    stats = {
        "trained_at"      : datetime.now().isoformat(),
        "model_type"      : "Bidirectional LSTM",
        "architecture"    : "BiLSTM(128) → LSTM(256) → Dense(128) → Dense(64) → Softmax",
        "accuracy"        : float(acc),
        "classes"         : list(le.classes_),
        "n_classes"       : len(gestures),
        "sequence_length" : SEQUENCE_LENGTH,
        "n_features"      : N_FEATURES,
        "total_params"    : model.count_params(),
        "train_samples"   : len(X_tr),
        "test_samples"    : len(X_te),
        "best_val_acc"    : float(max(history.history['val_accuracy'])),
        "epochs_run"      : len(history.history['loss']),
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  ✓ Model   → {MODEL_PATH}")
    print(f"  ✓ Encoder → {ENCODER_PATH}")
    print(f"  ✓ Stats   → {STATS_PATH}")
    print(f"\n  🎯 Final Accuracy: {acc*100:.2f}%")

    if acc < 0.85:
        print("\n  ⚠️  Accuracy below 85%. Tips:")
        print("     • Collect more sequences (aim 300+ per gesture)")
        print("     • Ensure consistent lighting and background")
        print("     • Perform gestures more deliberately")
    elif acc < 0.92:
        print("\n  ✅ Good accuracy! Consider collecting more data for 95%+")
    else:
        print("\n  🏆 Excellent accuracy! Model is ready for production.")

if __name__ == "__main__":
    train()
