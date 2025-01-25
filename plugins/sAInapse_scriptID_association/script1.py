import os
import matplotlib
import pathlib
import mne

def create_file():
    # Create the file and write the content
    with open("/app/cat/data/outcome_data/test.txt", "w") as file:
        file.write("color a car yellow")

def first_visualization():
    matplotlib.use('Qt5Agg')
    sample_data_dir = mne.datasets.sample.data_path()

    # Convert to a pathlib.Path for more convenience
    sample_data_dir = pathlib.Path(sample_data_dir)


    raw_path = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw(raw_path)
    

    raw.plot()


# Execute the create_file function only when this file is executed directly
if __name__ == "__main__":
    create_file()
