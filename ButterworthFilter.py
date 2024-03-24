import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Read the WAV file
samplerate, data = wavfile.read('test/std.wav')

# Apply Butterworth filter to denoise the audio
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
cutoff = 1200  # desired cutoff frequency of the filter, Hz
# Filter the data, and plot both the original and filtered signals.
filtered_data = butter_lowpass_filter(data, cutoff, samplerate, order)
# Save the filtered data to a new WAV file
wavfile.write('test/filtered_std.wav', samplerate, filtered_data.astype(np.int16))