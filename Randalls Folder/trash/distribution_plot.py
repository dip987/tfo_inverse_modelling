import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#DATA_PATH = r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/processed_data/processed1_min_long_range.pkl'
DATA_PATH = r'/home/rlfowler/Documents/research/tfo_inverse_modelling/Randalls Folder/data/rishad_data_ACDC.pkl'
#DATA_PATH = r'/home/rlfowler/Documents/research/tfo_inverse_modelling/Randalls Folder/data/rishad_data_intensities.pkl'

df = pd.read_pickle(DATA_PATH)
df.iloc[:, 7:] = df.iloc[:, 7:].abs()


def plot_distribution(df, detectors):
    # Set the number of rows based on the number of detectors
    nrows = len(detectors)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(26, 3 * nrows))

    # Loop through each detector and plot the distribution for each wavelength
    for i, detector in enumerate(detectors):
        # Wavelength 1
        sns.histplot(df, x=f'MAX_ACbyDC_WV1_{detector}', hue='Maternal Wall Thickness', stat='probability', line_kws={'color': 'crimson', 'lw': 2}, log_scale=True, bins=150, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'Detector {detector} - 740nm')
        axes[i, 0].set_xlabel('Pulsation Ratio')

        # Wavelength 2
        sns.histplot(df, x=f'MAX_ACbyDC_WV2_{detector}', hue='Maternal Wall Thickness', stat='probability', line_kws={'color': 'crimson', 'lw': 2}, log_scale=True, bins=150, kde=True, ax=axes[i, 1])
        axes[i, 1].set_title(f'Detector {detector} - 850nm')
        axes[i, 1].set_xlabel('Pulsation Ratio')

    # Adjust layout
    plt.tight_layout()
    plt.show()

plt.ion()
detectors = [1, 5, 8, 13, 19]  # Specify the detectors you want to plot , 5, 8, 13, 19
plot_distribution(df, detectors)

pass