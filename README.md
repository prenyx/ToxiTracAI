# ToxiTracAI  

ToxiTracAI ist ein Schulprojekt zur Entwicklung eines Machine-Learning-Systems zur Erkennung von Alkoholkonsum anhand von Herzfrequenzdaten. Das Projekt umfasst die Datenaufbereitung, Modelltraining und Anwendung durch ein neuronales Netzwerk.

## Inhaltsverzeichnis  

- [Überblick](#überblick)  
- [Projektstruktur](#projektstruktur)  
- [Installation](#installation)  
- [Anwendung](#anwendung)  
- [Hinweise zur Daten- und Modellorganisation](#hinweise-zur-daten--und-modellorganisation)  
- [Weiterentwicklungen](#weiterentwicklungen)  

---

## Überblick  

In diesem Projekt untersuchen wir Herzfrequenzdaten von alkoholisierten und nicht alkoholisierten Personen und erstellen ein Machine-Learning-Modell zur Erkennung von Herzfrequenzveränderungen.  
### Features:  
- Datenaufbereitung: Konsolidierung und Verarbeitung von Rohdaten.  
- Modelltraining: Neuronales Netzwerk für Klassifizierung normaler und abnormaler Zustände.  
- Einfache Anwendung via `main.py`.  

---

## Projektstruktur  

Hier ist ein Überblick über die Struktur des Projekts:  

```
ToxiTracAI/  
├── datasets/                                  # Roh- und verarbeitete Datensätze  
├── model-training/                            # Notebooks für Modelltraining  
├── models/                                    # Modell- und Scaler-Dateien  
│   ├── heart_nn_normalrate_model2.keras       # Trainiertes Modell  
│   └── scaler_nn_model2.joblib                # Scaler für Daten-Normalisierung  
├── wavelet-transform/                         # Experimente zur Datenanalyse  
├── .gitignore                                 # Ignorierte Dateien und Ordner  
├── config.py                                  # Zentrale Konfigurationsdatei für Pfade  
├── main.py                                    # Haupt-Skript zur Anwendung  
├── toxitracai.py                              # Hauptklasse für die Analyse  
├── README.md                                  # Projekt-Beschreibung  
└── requirements.txt                           # Abhängigkeiten für das Projekt  
```

---

## Installation  

1. **Repository klonen:**  
   ```bash
   git clone https://github.com/prenyx/ToxiTracAI.git
   cd ToxiTracAI
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Für Mac/Linux
   venv\Scripts\activate      # Für Windows
   ```

3. **Erforderliche Pakete installieren:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Anwendung  

1. **Datenaufbereitung:**  
   Im Notebook `data_preparation.ipynb` wurden die Herzfrequenzdaten von alkoholisierten und nicht alkoholisierten Personen zusammengeführt. Diese Daten sind jedoch noch nicht vollständig vorbereitet; es werden nur die **normalen BPM-Daten** verwendet.  

2. **Modellanwendung:**  
   Um das Programm auszuführen, starten Sie einfach die Datei `main.py`. Diese initialisiert die **ToxitracAI-Klasse**, die das vorbereitete Modell lädt und Vorhersagen über die eingegebenen BPM-Daten trifft.  
   ```bash
   python main.py
   ```

3. **Beispielausgabe:**  
   - Geben Sie minimale und maximale BPM-Werte sowie die aktuelle Bedingung (z. B. „resting“) ein.  
   - Das Modell gibt zurück, ob ein Zustand „active“ oder „normal“ erkannt wird.  

---

## Hinweise zur Daten- und Modellorganisation  

1. **Modelldateien:**  
   Die Modelldateien befinden sich im Ordner `models/`, darunter das trainierte Modell (`heart_nn_normalrate_model2.keras`) und der Scaler (`scaler_nn_model2.joblib`).  
   Alle Modellpfade werden über die Datei `config.py` dynamisch konfiguriert, sodass sie unabhängig vom jeweiligen System genutzt werden können.  

2. **Dataset-Ordner:**  
   Der Ordner `datasets/` enthält einige Beispiel-Datensätze, die zur Evaluierung des Modells verwendet werden können.  

3. **Wavelet-Daten:**  
   Im Ordner `wavelet-transform/` befinden sich experimentelle EKG-Daten für **normale BPM-Werte**, die in zukünftigen Iterationen des Projekts erweitert werden könnten.

4. **Model-Training:**  
   Der Ordner `model-training/` enthält alle relevanten Codes und Notebooks für das Training des neuronalen Netzwerks, einschließlich Datenaufteilung, Feature-Scaling und Modelloptimierung.

---

## Weiterentwicklungen  

- Integration der alkoholisierten Daten ins Modelltraining.  
- Validierung des Modells anhand eines größeren und diversifizierten Datensatzes.  
- Optimierung des neuronalen Netzwerks zur Verbesserung der Genauigkeit.  

---

### Viel Erfolg bei der Nutzung von ToxiTracAI! 🎉  

---
