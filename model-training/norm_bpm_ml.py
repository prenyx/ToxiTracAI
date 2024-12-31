import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pywt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Load your data
data = pd.read_csv('../datasets/bpm_norm_rates100.csv')


def map_condition(condition):
    """A function that map different conditions to normal or active"""
    if condition in ['normal', 'resting']:
        return 'normal'
    else:
        return 'active'


# Apply and encode the condition categories
data['Category'] = data['Condition'].apply(map_condition)
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Convert the 'Date' column to datetime and extract components
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Process Time Range into datetime
data[['start_time', 'end_time']] = data['Time Range'].str.split('-', expand=True)
data['start_time'] = pd.to_datetime(data['start_time'], format='%H:%M')
data['end_time'] = pd.to_datetime(data['end_time'], format='%H:%M')
data['duration_hours'] = (data['end_time'] - data['start_time']).dt.seconds / 3600

# Convert BPM values to numerics and normalize
data['BPM min'] = pd.to_numeric(data['BPM min'], errors='coerce')
data['BPM max'] = pd.to_numeric(data['BPM max'], errors='coerce')
scaler = StandardScaler()
data[['BPM min', 'BPM max']] = scaler.fit_transform(data[['BPM min', 'BPM max']])
dump(scaler, '../models/scaler.joblib')
print('Scaler saved')

# First, ensure 'BPM min' and 'BPM max' do not contain NaNs for DWT transformation
mask = data['BPM min'].notna() & data['BPM max'].notna()
# print(f'Display mask data:\n{mask.head()}')
data_filtered = data[mask]
print(f'Display filtered data:\n{data_filtered.head()}')

# Recalculate DWT features on filtered data
cA_min, _ = pywt.dwt(data_filtered['BPM min'].values, 'db1')
cA_max, _ = pywt.dwt(data_filtered['BPM max'].values, 'db1')

data_filtered = data_filtered.iloc[:len(cA_min)]  # Adjust the dataframe to match the DWT output length

# Ensure that the number of samples in your features and labels are identical
# Now prepare your feature set
X_filtered = data_filtered[['BPM min', 'BPM max']]
X_stacked = np.column_stack((X_filtered, cA_min, cA_max))
y_filtered = data_filtered['Category']

# Now, X_stacked and y_filtered can be used for splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_stacked, y_filtered, test_size=0.4, random_state=42)

print(X_train.shape)
print(X_test.shape)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(f'Classification report:\n{classification_report(y_test, predictions)}')

print("Training complete.")

# Save the model to a file
dump(model, '../models/heart_normalrate_model.joblib')
print('Saved model to heart_normalrate_model.joblib')
