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
recAmr = get_relative_path('Amrs recording.wav')
recMahmoud = get_relative_path('Mahmouds recording.wav')

def record_audio(duration, samplerate):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Recording finished.")
    return recording.flatten()

def plot_audio_signals(filepaths):
    plt.figure(figsize=(12, 6))
    
    for idx, filepath in enumerate(filepaths):
        data, samplerate = sf.read(filepath)
        
        # Ensure data is one-dimensional (single channel)
        if data.ndim > 1:
            data = data.mean(axis=1)  # Average the channels to create a single channel
        
        time = np.linspace(0., len(data) / samplerate, num=len(data))
        plt.plot(time, data, label=f'Audio {idx+1}')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time Signals of Recordings')
    plt.legend()
    plt.show()
    
def plot_audio(data, samplerate):
    plt.figure(figsize=(12, 6))
        
    time = np.linspace(0., len(data) / samplerate, num=len(data))
    plt.plot(time, data, label=f'Audio')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time Signals of Recordings')
    plt.legend()
    plt.show()

def scale_and_shift_signal(filepath, a, t0):
    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel
    
    t = np.arange(len(data)) / samplerate  # Original time axis
    t_scaled_shifted = (t* a - t0) # Apply scaling and shifting
    
    # Interpolation
    y = np.interp(t, t_scaled_shifted, data, left=0, right=0)
    
    return y, samplerate


def sigBefore(filepath):
    data, samplerate = sf.read(filepath)
    return np.arange(len(data)) / samplerate

def play_audio(data, samplerate):
    sd.play(data, samplerate)
    sd.wait()

def add_and_play_signals(filepath, a, t0):
    original_data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if original_data.ndim > 1:
        original_data = original_data.mean(axis=1)  # Average the channels to create a single channel
    
    scaled_shifted_data, _ = scale_and_shift_signal(filepath, a, t0)
    plot_audio(scaled_shifted_data, 48000)
    min_length = min(len(original_data), len(scaled_shifted_data))
    combined_signal = original_data[:min_length] + scaled_shifted_data[:min_length]
    play_audio(scaled_shifted_data, samplerate)

def plot_fourier_transform(filepath):
    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    
    plt.figure(figsize=(24, 6))
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform of the Signal')
    plt.show()

def plot_fourier_data(xf, yf):
    plt.figure(figsize=(24, 6))
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform of the Signal')
    plt.show()


def shift_frequency_and_play(filepath, freq_shift):
    print("shift frequency")
    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf_shifted = np.roll(yf, int(freq_shift * N / samplerate))
    shifted_signal = ifft(yf_shifted)
    play_audio(shifted_signal.real, samplerate)

def low_pass_filter_and_play(filepath, cutoff_freq):
    print("low pass:")
    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel
    
    

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf[np.abs(xf) > cutoff_freq] = 0
    plot_fourier_data(xf,yf )
    filtered_signal = ifft(yf)
    play_audio(filtered_signal.real, samplerate)

def high_pass_filter_and_play(filepath, cutoff_freq):
    print("high pass:")

    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    yf[np.abs(xf) < cutoff_freq] = 0
    plot_fourier_data(xf,yf )

    filtered_signal = ifft(yf)
    play_audio(filtered_signal.real, samplerate)

def triangular_filter_and_play(filepath, wc):
    print("triangular filter")
    data, samplerate = sf.read(filepath)
    
    # Ensure data is one-dimensional (single channel)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Average the channels to create a single channel

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
    #plot_fourier_data(xf, yf_filtered)
    filtered_signal = ifft(yf_filtered)
    play_audio(filtered_signal.real, samplerate)
    print("yf_filtered")
    plot_fourier_data(xf, yf_filtered)
    print("yf")
    plot_fourier_data(xf, yf)
    print("")
    # Plot the filter response
    plt.figure(figsize=(12, 6))
    plt.plot(xf, filter_response)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Triangular Filter Response')
    plt.show()

def main():
    choice = input("Do you want to (1) record your own sound or (2) use pre-existing sounds? Enter 1 or 2: ")

    if choice == '1':
        duration = float(input("Enter the duration of the recording in seconds: "))
        samplerate = 48000
        recording = record_audio(duration, samplerate)
        filepath = 'user_recording.wav'
        sf.write(filepath, recording, samplerate)
    elif choice == '2':
        print("Choose from the following pre-existing sounds:")
        print("1. Ahmed's recording")
        print("2. Amr's recording")
        print("3. Mahmoud's recording")
        sound_choice = input("Enter 1, 2, or 3: ")
        if sound_choice == '1':
            filepath = recAhmed
        elif sound_choice == '2':
            filepath = recAmr
        elif sound_choice == '3':
            filepath = recMahmoud
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    filepaths = [filepath]
    
    # # Plot audio signals
    # plot_audio_signals(filepaths)

    # # Scale and shift signal, then add and play
    # add_and_play_signals(filepath,1,1.5 )

    # # Plot Fourier transform
    # plot_fourier_transform(filepath)

    # # Shift frequency and play
    # shift_frequency_and_play(filepath, 200)

    # # Apply low pass filter and play
    # low_pass_filter_and_play(filepath, 3000)

    # # Apply high pass filter and play
    # high_pass_filter_and_play(filepath, 3000)

    # Apply triangular filter and play
    triangular_filter_and_play(filepath, 3000)

if __name__ == "__main__":
    main()
