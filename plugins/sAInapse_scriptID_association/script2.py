import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
import io
import base64

def load_mat_files():
    mat_files = []
    for root, dirs, files in os.walk("/app/cat/data/datasets/s1"):
        for f in files:
            if f.endswith(".mat"):
                mat_files.append(os.path.join(root, f))

    if not mat_files:
        raise FileNotFoundError("No .mat files found. Check the unzipped structure.")

    demo_mat = mat_files[0]
    print(f"\nUsing: {demo_mat}")

    data = loadmat(demo_mat)
    print("Keys in .mat:", list(data.keys()))

    emg_signal = data['emg']
    labels = data['restimulus'].flatten()

    print("EMG shape:", emg_signal.shape)
    print("Unique labels:", np.unique(labels))

    plt.figure(figsize=(12,4))
    plt.plot(emg_signal[:1000, 0], label="Channel 0")
    plt.title("First 1000 Samples of EMG Channel 0")
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
            <h1>Plot of raw data</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/raw_data.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")


if __name__ == "__main__":
    load_mat_files()