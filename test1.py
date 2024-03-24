import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import librosa

# Function to apply Butterworth filter
def butterworth_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to segment the signal into action parts
def segment_signal(signal, window_size, step_size):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segment = signal[start:start+window_size]
        segments.append(segment)
    return segments

# Function to extract characteristics from each segment
def extract_features(segment):
    # Example feature extraction (you can customize this according to your needs)
    mean_amp = np.mean(segment)
    std_amp = np.std(segment)
    max_amp = np.max(segment)
    min_amp = np.min(segment)
    return [mean_amp, std_amp, max_amp, min_amp]

# Load ultrasonic data (replace 'filename.wav' with your file)
filename = 'filename.wav'
signal, fs = librosa.load(filename, sr=None)

# Preprocess the signal (example: apply Butterworth filter)
lowcut = 1000  # Define low cutoff frequency
highcut = 5000  # Define high cutoff frequency
order = 6  # Define filter order
filtered_signal = butterworth_filter(signal, lowcut, highcut, fs, order=order)

# Segment the signal into action parts
window_size = int(fs * 0.1)  # Define window size (e.g., 0.1 seconds)
step_size = int(fs * 0.05)  # Define step size (e.g., 0.05 seconds)
segments = segment_signal(filtered_signal, window_size, step_size)

# Extract characteristics from each segment
features = [extract_features(segment) for segment in segments]  # List comprehension

# Create a DataFrame to store the extracted features
columns = ['Mean Amplitude', 'Standard Deviation', 'Max Amplitude', 'Min Amplitude']
df = pd.DataFrame(features, columns=columns)

# Add labels for each action (you can customize this)
action_labels = ['Action1', 'Action2', 'Action3', ...]  # Add labels for each action
action_labels *= len(segments) // len(action_labels) + 1  # Repeat labels if necessary
df['Action'] = action_labels[:len(df)]

# Save the DataFrame to a CSV file
df.to_csv('extracted_features.csv', index=False)

