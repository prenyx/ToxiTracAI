from joblib import load
import pandas as pd
import tensorflow as tf
import os

KERAS_FILE_PATH = 'C:/Users/sthee/SynologyDrive/PycharmProjects/ToxitracAI/models/heart_nn_normalrate_model2.keras'
SCALER_FILE_PATH = 'C:/Users/sthee/SynologyDrive/PycharmProjects/ToxitracAI/models/scaler_nn_model2.joblib'


class ToxitracAI:

    def __init__(self, user, bpm_min, bpm_max, condition):
        self._user = user
        self._bpm_min = bpm_min
        self._bpm_max = bpm_max
        self._condition = condition

        # Load trained model
        self.model = tf.keras.models.load_model(KERAS_FILE_PATH)
        self.scaler = load(SCALER_FILE_PATH)

    # @tf.function(reduce_retracing=True)
    def predict_condition(self):
        """A function to predict its condition based on bpm features"""

        # Define thresholds for BPM rates based on condition
        high_bpm_threshold = 100 if self._condition in ['resting', 'normal'] else 200
        low_bpm_threshold = 60 if self._condition in ['resting', 'normal'] else 100
        sleep_bpm_threshold = 40 if self._condition in ['resting'] else 50

        # Check if BPM max exceeds high threshold
        if self._bpm_max > high_bpm_threshold:
            print(f'Warning: Your BPM rate {self._bpm_max} is unusually high for the condition {self._condition}!'
                  f" Please monitor your BPM rate regularly, and consider consulting a healthcare professional if "
                  f"this high BPM persists or if you experience any discomfort.")

        # Check if BPM min is in the sleep threshold range
        elif sleep_bpm_threshold <= self._bpm_min < low_bpm_threshold:
            print(f'Note: Your BPM rate {self._bpm_min} is very low, which may be typical during sleep. If this is unexpected, consider monitoring it further.')

        # Check if BPM min is below low threshold and not in sleep range
        elif self._bpm_min < sleep_bpm_threshold:
            print(f'Warning: Your BPM rate {self._bpm_min} is unusually low for the condition {self._condition}!'
                  f" Please monitor your BPM rate regularly, and consult a healthcare professional if this low BPM persists.")

        # Prepare input with feature names
        input_features_df = pd.DataFrame([[self._bpm_min, self._bpm_max]], columns=['BPM min', 'BPM max'])
        input_scaled = self.scaler.transform(input_features_df)

        # Convert to tensor for model input
        input_scaled_tensor = tf.convert_to_tensor(input_scaled, dtype=tf.float32)

        # Make prediction
        prediction = self.model(input_scaled_tensor)
        condition = 'active' if tf.argmax(prediction, axis=1).numpy()[0] == 1 else 'normal'
        return condition

    @property
    def user(self):
        return self._user

    @property
    def bpm_min(self):
        return self._bpm_min

    @property
    def bpm_max(self):
        return self._bpm_max

    @property
    def condition(self):
        return self.condition

    @user.setter
    def user(self, name):
        self._user = name
        return self._user

    @bpm_min.setter
    def bpm_min(self, bpm_min):
        if bpm_min <= 0:
            raise ValueError("BPM min cannot be less than or equal to zero")
        else:
            self._bpm_min = bpm_min

    @bpm_max.setter
    def bpm_max(self, bpm_max):
        if bpm_max <= 0:
            raise ValueError("BPM max cannot be less than or equal to zero")
        else:
            self._bpm_max = bpm_max

    @condition.setter
    def condition(self, condition):
        valid_conditions = ['resting', 'normal', 'exercising', 'after workout']
        if condition not in valid_conditions:
            raise ValueError('Please enter a valid condition like "resting", "normal", "exercising" or "after workout"')
        else:
            self._condition = condition