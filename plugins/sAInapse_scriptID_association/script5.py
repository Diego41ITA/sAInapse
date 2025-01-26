import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
import base64
import os
from scipy.io import loadmat
import requests
import zipfile

# Parametri
window_size = 200  # Dimensione della finestra temporale
num_classes = 23  # Numero di classi (gesti)
num_channels = 16  # Numero di canali EMG

def run_cnn():
    base_url = "https://ninapro.hevs.ch/files/DB5_Preproc/"
    subjects = [f"s{i}" for i in range(1, 11)]  # Soggetti da s1 a s10

    all_emg_signals = []
    all_labels = []

    for subject in subjects:
        zip_filename = f"{subject}.zip"
        extract_folder = f"ninapro_{subject}"

        # Scaricare il file .zip
        if not os.path.exists(zip_filename):
            print(f"Downloading {zip_filename}...")
            url = base_url + zip_filename
            r = requests.get(url, allow_redirects=True)
            open(zip_filename, 'wb').write(r.content)
            print(f"{zip_filename} downloaded.")
        else:
            print(f"{zip_filename} already exists, skipping download.")

        # Estrarre i file
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            zf.extractall(extract_folder)

        # Elencare i file .mat
        dataset_folder = f"{extract_folder}/{subject}"
        mat_files = [
            os.path.join(root, f)
            for root, dirs, files in os.walk(dataset_folder)
            for f in files if f.endswith(".mat")
        ]

        if not mat_files:
            raise FileNotFoundError(f"No .mat files found for {subject}.")

        # Processare ogni file .mat
        for mat_file in mat_files:
            data = loadmat(mat_file)

            # Estrarre segnali EMG e etichette
            emg_signal = data['emg']  # Segnali EMG
            labels = data['restimulus'].flatten()  # Etichette dei gesti

            # Filtrare i dati validi (escludere etichette 0)
            valid_idx = labels > 0
            emg_signal = emg_signal[valid_idx]
            labels = labels[valid_idx]
            
            # Accumulare i dati
            all_emg_signals.append(emg_signal)
            all_labels.append(labels)

    # Concatenare tutti i dati
    all_emg_signals = np.vstack(all_emg_signals)
    all_labels = np.concatenate(all_labels)

    print("Dataset completo:")
    print("EMG shape:", all_emg_signals.shape)
    print("Etichette uniche:", np.unique(all_labels))

    # Caricamento del dataset (assumiamo che sia già caricato come X e y)
    # X = ...  # Shape: (2384532, 16)
    # y = ...  # Shape: (2384532,)

    X = all_emg_signals
    y = all_labels


    # Normalizzazione dei dati
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)  # Normalizziamo per ogni canale (colonna)

    # Creazione di finestre temporali
    def create_windows(X, y, window_size):
        windows = []
        labels = []
        for i in range(0, len(X) - window_size, window_size):  # Creiamo finestre non sovrapposte
            windows.append(X[i:i+window_size])  # Ogni finestra contiene 'window_size' campioni
            labels.append(y[i + window_size - 1])  # L'etichetta è quella dell'ultimo campione della finestra
        return np.array(windows), np.array(labels)

    # Creazione delle finestre
    X_windows, y_windows = create_windows(X_normalized, y, window_size)

    # Riformattare la shape per la rete neurale (samples, time_steps, channels)
    X_windows = X_windows.reshape(X_windows.shape[0], window_size, num_channels)

    # Codifica delle etichette (one-hot encoding)
    y_windows_encoded = to_categorical(y_windows-1, num_classes=num_classes)

    # Creazione del modello
    model = Sequential()

    # Strato di convoluzione 1
    model.add(Conv1D(32, 5, activation='relu', input_shape=(window_size, num_channels), kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D(2))

    # Strato di convoluzione 2
    model.add(Conv1D(64, 5, activation='relu',  kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D(2))

    # Flattening
    model.add(Flatten())

    # Strato completamente connesso
    model.add(Dense(128, activation='relu',  kernel_regularizer=l2(0.01)))
    #model.add(Dropout(0.5))  # Dropout per evitare overfitting

    # Strato di output
    model.add(Dense(num_classes, activation='softmax'))  # 23 classi per i gesti

    # Compilazione del modello
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Riassunto del modello
    model.summary()


    # Aggiunta dell'EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitora la perdita di validazione
                                patience=10,           # Se non ci sono miglioramenti per 5 epoche, interrompe l'allenamento
                                restore_best_weights=True)  # Ripristina i pesi migliori quando l'allenamento si interrompe

    # Divisione dei dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows_encoded, test_size=0.2, random_state=42)

    # Addestramento del modello
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Valutazione del modello
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_acc:.4f}')

    import matplotlib.pyplot as plt

    # Grafico della loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
            <h1>Plot of training and validation loss</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/training_validation_loss.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")


    # Grafico dell'accuratezza
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

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
            <h1>Plot of training and validation accuracy</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/training_validation_accuracy.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Ottieni la classe con probabilità massima

    # Calcolare la matrice di confusione
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

    # Creare la heatmap della matrice di confusione
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, num_classes+1), yticklabels=np.arange(1, num_classes+1))
    plt.xlabel('Classe Predetta')
    plt.ylabel('Classe Reale')
    plt.title('Matrice di Confusione - Heatmap')
    plt.show()

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
            <h1>Plot of confusion matrix</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/confusion_matrix.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")

if __name__ == "__main__":
    run_cnn()