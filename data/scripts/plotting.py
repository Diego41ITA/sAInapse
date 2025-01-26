import matplotlib.pyplot as plt
import numpy
from scipy.io import loadmat
import os

# Walk the .mat files inside the unzipped folder
mat_files = []
for root, dirs, files in os.walk("/app/cat/data/datasets/s1"):
    for f in files:
        if f.endswith(".mat"):
            mat_files.append(os.path.join(root, f))

if not mat_files:
    raise FileNotFoundError("No .mat files found. Check the unzipped structure.")

print("Found .mat files:")
for mf in mat_files:
    print("  ", mf)

demo_mat = mat_files[0]
print(f"\nUsing: {demo_mat}")

data = loadmat(demo_mat)
print("Keys in .mat:", list(data.keys()))

# Suppose we have 'emg' and 'restimulus' keys:
emg_signal = data['emg']  # shape (N, 16)
labels = data['restimulus'].flatten()  # shape (N, ), integer-coded

print("EMG shape:", emg_signal.shape)
print("Unique labels:", numpy.unique(labels))

plt.figure(figsize=(12,4))
plt.plot(emg_signal[:1000, 0], label="Channel 0")
plt.title("First 1000 Samples of EMG Channel 0")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()   
