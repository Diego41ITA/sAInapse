import zipfile
import requests
import os
import numpy as np
from scipy.io import loadmat
import io
import base64
import matplotlib.pyplot as plt

def create_file():
    # Create the file and write the content
    with open("/app/cat/data/outcome_data/test.txt", "w") as file:
        file.write("color a car yellow")

# def first_visualization():
#     #matplotlib.use('Qt5Agg')
#     sample_data_dir = mne.datasets.sample.data_path()

#     # Convert to a pathlib.Path for more convenience
#     sample_data_dir = pathlib.Path(sample_data_dir)

#     print(sample_data_dir)

#     raw_path = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
#     raw = mne.io.read_raw(raw_path)
    

#     raw.plot()

def downlaod_dataset():
# URL of the zip file
    url = "https://ninapro.hevs.ch/files/DB5_Preproc/s1.zip"  # Example for s1

    # Specify the folder where you want to download the zip file
    download_folder = "/app/cat/data/datasets"  # Change this to your desired folder
    os.makedirs(download_folder, exist_ok=True)  # Make sure the folder exists


    zip_filename = "s1.zip"

    # Download the file if it doesn't already exist
    if not os.path.exists(zip_filename):
        print(f"Downloading {url} to {zip_filename} ...")
        r = requests.get(url, allow_redirects=True)
        with open(zip_filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"{zip_filename} already present, skipping download.")

    # Specify the folder where you want to extract the contents
    extract_folder = "/app/cat/data/datasets"  
    os.makedirs(extract_folder, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zf:
        zf.extractall(extract_folder)

    print(f"Files extracted to {extract_folder}")

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

    # plt.figure(figsize=(12,4))
    # plt.plot(emg_signal[:1000, 0], label="Channel 0")
    # plt.title("First 1000 Samples of EMG Channel 0")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.legend()

    # # Prova a scrivere direttamente su BytesIO
    # img_bytes = io.BytesIO()
    # plt.savefig(img_bytes, format='png')
    # img_bytes.seek(0)

    # if img_bytes.getvalue():
    #     print("L'immagine è stata scritta correttamente in memoria.")
    # else:
    #     print("Errore nella scrittura dell'immagine in memoria.")
    #     return

    # img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # html_response = f'''
    # <html>
    #     <body>
    #         <h1>Grafico generato con successo</h1>
    #         <img src="data:image/png;base64,{img_base64}" />
    #     </body>
    # </html>
    # '''

    # # Salva il file HTML fuori dal container, se vuoi
    # output_path = "/app/cat/data/plots/grafico.html"
    
    # try:
    #     with open(output_path, "w") as f:
    #         f.write(html_response)
    #         print(f"File {output_path} scritto con successo.")
    # except Exception as e:
    #     print(f"Errore nel salvataggio del file: {e}")


# Execute the create_file function only when this file is executed directly
if __name__ == "__main__":
    create_file()
