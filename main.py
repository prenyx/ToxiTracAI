from toxitracai import ToxitracAI
import os


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow messages
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable optimizations, including AVX2


NORM_DATA_FILE_PATH = 'C:/Users/sthee/SynologyDrive/PycharmProjects/ToxitracAI/bpm_norm_rates200.csv'


def main():
    print("Welcome to Toxitrac AI - Your Alcohol Consumption Detection Assistant!\n")

    user = input("Who is running the analysis today?: ")
    bpm_min = float(input("Enter your BPM min: "))
    bpm_max = float(input("Enter your BPM max: "))
    condition = input("Enter your condition (e.g., resting, normal, exercising, after workout): ").strip().lower()

    try:
        toxitrac = ToxitracAI(user=user, bpm_min=bpm_min, bpm_max=bpm_max, condition=condition)
        # Predict condition
        prediction = toxitrac.predict_condition()
        print(f"{toxitrac.user}, your predicted condition is {prediction}.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
