FROM ghcr.io/cheshire-cat-ai/core:latest

# Install Python libraries
RUN pip install --no-cache-dir numpy pandas matplotlib pathlib mne

