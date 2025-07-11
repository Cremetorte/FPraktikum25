import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Funktion zum Einlesen der CSV-Datei mit Leerzeilen-Trennung ===
def read_datasets_from_csv(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    # Splitte nach doppelten Zeilenumbrüchen (Datensatztrenner)
    raw_datasets = content.strip().split('\n\n')

    datasets = []
    for raw in raw_datasets:
        from io import StringIO
        df = pd.read_csv(StringIO(raw), sep=',')
        datasets.append(df)
    return datasets

# === Heatmap-Plot erstellen ===
def plot_heatmap(datasets):
    num_datasets = len(datasets)
    frequencies = datasets[0]['#Frequency (Hz)'].values
    print(datasets[0].keys())
    s21db_matrix = np.array([ds['S21dB (dB)'].values for ds in datasets])

    fig, ax = plt.subplots()
    c = ax.imshow(s21db_matrix, aspect='auto', extent=[frequencies[0], frequencies[-1], num_datasets - 0.5, -0.5],
                  cmap='viridis', origin='upper')
    ax.set_xlabel('Frequenz (Hz)')
    ax.set_ylabel('Datensatz-Nummer')
    fig.colorbar(c, ax=ax, label='S21dB (dB)')
    ax.set_title('S21dB Heatmap')

    def onclick(event):
        if event.inaxes != ax:
            return
        # Berechne die nächstliegende Datensatznummer
        dataset_index = int(round(event.ydata))
        if 0 <= dataset_index < num_datasets:
            plot_single_dataset(datasets[dataset_index], dataset_index)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# === Einzelnen Datensatz plotten ===
def plot_single_dataset(dataset, index):
    plt.figure()
    plt.plot(dataset['#Frequency (Hz)'], dataset['S21dB (dB)'])
    plt.title(f'Datensatz {index}: S21dB vs Frequenz')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('S21dB (dB)')
    plt.grid(True)
    plt.show()

# === Hauptprogramm ===
if __name__ == '__main__':
    filepath = 'Data/Group_041_2025_07_01_14.55.45_n3_VNA_res1_lin_20_3.24GHz/Group_041_2025_07_01_14.55.45_n3_VNA_res1_lin_20_3.24GHz.dat'  # <-- Pfad zu deiner Datei hier anpassen
    datasets = read_datasets_from_csv(filepath)
    plot_heatmap(datasets)
