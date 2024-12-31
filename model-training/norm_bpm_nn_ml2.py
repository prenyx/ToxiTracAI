import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from joblib import dump
from keras.callbacks import EarlyStopping


def map_condition(condition):
    return 0 if condition in ['normal', 'resting'] else 1


# Load and prepare data
data = pd.read_csv('../datasets/bpm_norm_rates200.csv')
data['Category'] = data['Condition'].apply(map_condition)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(data[['BPM min', 'BPM max']])
y = tf.keras.utils.to_categorical(data['Category'])

# Save the scaler to a file
dump(scaler, '../models/scaler_nn_model2.joblib')
print('Scaler saved successfully into "scaler_nn_model2.joblib"')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Neural Network Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with EarlyStopping
history = model.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    shuffle=True
)

# Save your model
keras.saving.save_model(model, '../models/heart_nn_normalrate_model2.keras')
# model.save('heart_nn_normalrate_model.h5')
print('heart_nn_normalrate_model2.keras')

# Predictions (Evaluate Model)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification report
print(classification_report(y_test_labels, y_pred_labels))