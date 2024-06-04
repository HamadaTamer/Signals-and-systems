import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from os import path
from scipy.fft import fft, ifft, fftfreq

# Ensure cross-platform compatibility for file paths
def get_relative_path(filename):
    return path.join(path.dirname(__file__), filename)

# Example file path
recAhmed = get_relative_path('Ahmeds recording.wav')

def plot_audio_signals(filepaths):
    plt.figure(figsize=(12, 6))
    
    for idx, filepath in enumerate(filepaths):
        data, samplerate = sf.read(filepath)
        time = np.linspace(0., len(data) / samplerate, num=len(data))
        plt.plot(time, data, label=f'Audio {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time Signals of Recordings')
    plt.legend()
    plt.show()

def scale_and_shift_signal(filepath, a, t0, samplerate):
    data, original_samplerate = sf.read(filepath)
    if original_samplerate != samplerate:
        raise ValueError(f"Sampling rate mismatch: {original_samplerate} vs {samplerate}")

    t = np.arange(len(data)) / samplerate
    t_new = np.arange(0, len(data)) / (a * samplerate) + t0
    y = np.interp(t, t_new, data, left=0, right=0)
    
    return y

def sigBefore(filepath):
    samplerate = 48000
    data, original_samplerate = sf.read(filepath)
    if original_samplerate != samplerate:
        raise ValueError(f"Sampling rate mismatch: {original_samplerate} vs {samplerate}")

    return (np.arange(len(data)) / samplerate)

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def add_and_play_signals(filepath, a, t0, samplerate):
    original_data, _ = sf.read(filepath)
    scaled_shifted_data = scale_and_shift_signal(filepath, a, t0, samplerate)
    min_length = min(len(original_data), len(scaled_shifted_data))
    combined_signal = original_data[:min_length] + scaled_shifted_data[:min_length]
    play_audio(combined_signal, samplerate)

def plot_fourier_transform(filepath, samplerate):
    data, _ = sf.read(filepath)
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    
    plt.figure(figsize=(24, 6))
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform of the Signal')
    plt.show()

def shift_frequency_and_play(filepath, freq_shift, samplerate):
    data, _ = sf.read(filepath)
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf_shifted = np.roll(yf, int(freq_shift * N / samplerate))
    shifted_signal = ifft(yf_shifted)
    play_audio(shifted_signal.real, samplerate)

def low_pass_filter_and_play(filepath, cutoff_freq, samplerate):
    data, _ = sf.read(filepath)
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf[np.abs(xf) > cutoff_freq] = 0
    filtered_signal = ifft(yf)
    play_audio(filtered_signal.real, samplerate)

def high_pass_filter_and_play(filepath, cutoff_freq, samplerate):
    data, _ = sf.read(filepath)
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf[np.abs(xf) < cutoff_freq] = 0
    filtered_signal = ifft(yf)
    play_audio(filtered_signal.real, samplerate)

def triangular_filter_and_play(filepath, wc, samplerate):
    data, _ = sf.read(filepath)
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    
    # Create the triangular filter
    filter_response = np.zeros_like(xf)
    for i in range(len(xf)):
        if -wc <= xf[i] <= -wc/2:
            filter_response[i] = (xf[i] + wc) / (wc / 2)
        elif -wc/2 < xf[i] <= wc/2:
            filter_response[i] = 1 - 2 * abs(xf[i]) / wc
        elif wc/2 < xf[i] <= wc:
            filter_response[i] = (wc - xf[i]) / (wc / 2)
    
    yf_filtered = yf * filter_response
    filtered_signal = ifft(yf_filtered)
    play_audio(filtered_signal.real, samplerate)
    
    # Plot the filter response
    plt.figure(figsize=(12, 6))
    plt.plot(xf, filter_response)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Triangular Filter Response')
    plt.show()

# Example usages
recAhmed = get_relative_path('Ahmeds recording.wav')
filepaths = [recAhmed]

# Plot audio signals
plot_audio_signals(filepaths)

# Scale and shift signal, then add and play
add_and_play_signals(recAhmed, 1.5, 0.5, 48000)

# Plot Fourier transform
plot_fourier_transform(recAhmed, 48000)

# Shift frequency and play
shift_frequency_and_play(recAhmed, 1000, 48000)

# Apply low pass filter and play
low_pass_filter_and_play(recAhmed, 3000, 48000)

# Apply high pass filter and play
high_pass_filter_and_play(recAhmed, 3000, 48000)

# Apply triangular filter and play
triangular_filter_and_play(recAhmed, 3000, 48000)
