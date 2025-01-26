from scipy.signal import butter
from scipy import signal
import base64
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import io

def create_filter():
    mat_files = []
    for root, dirs, files in os.walk("/app/cat/data/datasets/s1"):
        for f in files:
            if f.endswith(".mat"):
                mat_files.append(os.path.join(root, f))

    if not mat_files:
        raise FileNotFoundError("No .mat files found. Check the unzipped structure.")

    demo_mat = mat_files[0]

    data = loadmat(demo_mat)

    emg_signal = data['emg']

    # Parameters
    fsEMG = 1000  # Sampling frequency, for example 1000 Hz
    steps_EMG_1muscle_1step = emg_signal  # Example data, replace with your real data

    # Band-pass filter between 15 and 450 Hz
    lowcut = 15
    highcut = 450
    nyquist = 0.5 * fsEMG
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create the band-pass filter (Butterworth)
    b, a = signal.butter(2, [low, high], btype='bandpass')
    steps_EMG_1muscle_1step_filt = signal.filtfilt(b, a, steps_EMG_1muscle_1step)

    # Notch filter between 48 and 52 Hz (stop-band filter)
    lowstop = 48 / nyquist
    highstop = 52 / nyquist

    # Create the notch filter (Butterworth)
    sos = signal.butter(2, [lowstop, highstop], btype='bandstop', output='sos')
    

    # Apply the notch filter
    steps_EMG_1muscle_1step_filtered = signal.sosfiltfilt(sos, steps_EMG_1muscle_1step_filt)

    plt.figure(figsize=(12,4))
    plt.plot(steps_EMG_1muscle_1step_filtered[:1000, 0], label="Channel 0")
    plt.title("First 1000 Samples of filtered EMG Channel 0")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # Prova a scrivere direttamente su BytesIO
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    if img_bytes.getvalue():
        print("The image has been correctly saved in the memory.")
    else:
        print("Error in saving the image in the memory.")
        return

    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    html_response = f'''
    <html>
        <body>
            <h1>Plot of filtered data</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/filtered_data.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")


if __name__ == "__main__":
    create_filter()
