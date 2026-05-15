# Auralis

Un sistema di **attenzione selettiva uditiva** ispirato al *cocktail party problem*: non si limita a separare le voci in una scena multi-speaker, ma sceglie automaticamente *quale* parlante ascoltare, ricostruendone il segnale isolato.

---

## Il problema

Quando ci troviamo in un ambiente affollato — una festa, un bar, una sala riunioni — il nostro cervello compie un'operazione straordinaria: riesce a concentrarsi su una singola voce, ignorando il rumore di fondo e le altre conversazioni. Questo fenomeno è noto come **cocktail party problem**, formalizzato da Colin Cherry nel 1953.

I sistemi tradizionali di *source separation* affrontano il problema da una prospettiva puramente algoritmica: separare tutte le sorgenti audio presenti nel segnale. Manca però il passo successivo, quello veramente umano: **decidere a quale voce prestare attenzione**.

Questo progetto affronta entrambi i lati del problema:

- **Separazione** (DSP) — isolare le componenti acustiche dei diversi parlanti.
- **Attenzione selettiva** (AI) — scegliere automaticamente il parlante target sulla base delle sue caratteristiche acustiche (pitch, energia, timbro, genere vocale), simulando il meccanismo cognitivo umano.

---

## Architettura

```
                  ┌───────────────────────┐
   Audio mix ───► │  DSP: feature extract │  (STFT, MFCC, pitch, energia)
                  └───────────┬───────────┘
                              │ features
                              ▼
                  ┌───────────────────────┐
                  │  AI: attention module │  (classificatore parlante target)
                  └───────────┬───────────┘
                              │ target speaker ID / mask
                              ▼
                  ┌───────────────────────┐
                  │  DSP: separation +    │  (mask application, ISTFT,
                  │  reconstruction       │   speech enhancement)
                  └───────────┬───────────┘
                              ▼
                       Audio isolato
```

Il sistema è composto da due moduli che lavorano in pipeline:

1. **Modulo DSP** — analizza il segnale audio in ingresso ed estrae le feature acustiche fondamentali.
2. **Modulo AI** — riceve le feature, classifica i parlanti e seleziona il target di attenzione.
3. **Modulo DSP (ricostruzione)** — applica una maschera tempo-frequenza al segnale e ricostruisce l'audio del solo parlante target, con speech enhancement post-processing.

---

## Requisiti di sistema

- **Python 3.10+**
- **Sistema operativo:** Linux, macOS o Windows
- **RAM:** 4 GB minimi consigliati

### Dipendenze Python principali

- `numpy`, `scipy` — operazioni numeriche e DSP
- `librosa`, `soundfile` — audio I/O e feature extraction
- `scikit-learn` — modelli ML leggeri
- `noisereduce` — speech enhancement
- `matplotlib` — visualizzazione di spettrogrammi e diagnostica
- `pytest` — testing

Vedi `requirements.txt` per l'elenco completo e le versioni.

---

## Installazione

```bash
# 1. Clona il repository
git clone <repo-url>
cd auralis

# 2. Crea e attiva un ambiente virtuale
python -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt
```

---

## Utilizzo

### Demo rapida

```bash
python demo.py
```

Genera automaticamente un mix da LibriSpeech (voce F + voce M), esegue la pipeline completa e salva 4 file in `data/processed/demo/`: `mix.wav`, `target.wav`, `interferer.wav`, `output.wav`.

### Pipeline end-to-end su file esterno

```bash
python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
```

### Training del classificatore

```bash
python -m src.ai.train --n-samples 400 --snr-db -3.0 0.0 3.0 --clip-duration 4.0 --out models/classifier.joblib
```

### Test

```bash
pytest tests/
```

---

## Struttura del progetto

```
auralis/
├── CLAUDE.md                  # Memoria persistente per Claude Code
├── README.md                  # Questo file
├── ROADMAP.md                 # Piano d'azione e stato di avanzamento
├── requirements.txt           # Dipendenze Python
├── demo.py                    # Script di demo rapida
│
├── src/                       # Codice sorgente
│   ├── dsp/                   # Componente DSP
│   │   ├── stft.py            # STFT / ISTFT
│   │   ├── features.py        # 44 feature/frame: MFCC+delta+delta², pitch, RMS, ecc.
│   │   ├── dataset.py         # Caricamento LibriSpeech, mix M+F, IBM labels
│   │   ├── enhancement.py     # Speech enhancement (noisereduce + peak norm)
│   │   ├── separation.py      # Masking T-F: ratio mask + pitch mask + male suppression
│   │   └── nmf_separation.py  # Separazione NMF guidata dal classificatore
│   ├── ai/                    # Componente AI
│   │   ├── classifier.py      # SpeakerClassifier (Random Forest + StandardScaler)
│   │   ├── attention.py       # AttentionModule: maschera per frame
│   │   └── train.py           # Training IBM multi-SNR su mix LibriSpeech
│   ├── pipeline.py            # CLI end-to-end
│   └── utils.py
│
├── data/                      # Dati audio (raw/ + processed/)
├── models/                    # Modelli addestrati
├── notebooks/                 # Esperimenti e analisi
├── tests/                     # Test unitari
└── docs/                      # Documentazione teorica
```

---

## Come funziona

### Pipeline AI + DSP

#### 1. Pre-processing (DSP)
Il segnale audio in ingresso viene caricato, eventualmente ricampionato a una frequenza standard (es. 16 kHz) e segmentato in finestre temporali tramite **Short-Time Fourier Transform (STFT)**. Questo permette di lavorare in dominio tempo-frequenza.

#### 2. Feature extraction (DSP)
Per ogni finestra temporale vengono estratte 44 feature acustiche:
- **MFCC + delta + delta²** — timbro e sua evoluzione temporale (39 feature).
- **Pitch (F0)** — frequenza fondamentale nel range femminile (150–310 Hz) via pYIN.
- **Energia (RMS)** — intensità del segnale.
- **Spectral centroid / rolloff** — caratteristiche del colore timbrico.
- **ZCR** (Zero Crossing Rate) — indicatore della natura voiced/unvoiced.

#### 3. Attention module (AI)
Le feature vengono date in input a un **Random Forest** (100 alberi) + StandardScaler, addestrato su frame reali di mix a SNR variabile (-3, 0, +3 dB) con etichette IBM frame-level. Per ogni finestra il modello produce una probabilità "femminile" in [0, 1].

#### 4. Separation NMF (DSP)
La separazione avviene in `nmf_separation.py`:
1. **NMF**: decompone lo spettrogramma di magnitudine in K=8 componenti (V ≈ W×H).
2. **Dominant-frame scoring**: ogni componente NMF riceve uno score di "femminilità" basato sulle attention weights nei frame in cui quella componente è dominante.
3. **IRM ibrida**: l'IRM per bin è la media pesata tra le attention weights del classificatore (65%, segnale temporale affidabile) e la soft mask NMF lineare V_f/(V_f+V_m) (35%, risoluzione in frequenza).
4. **Pitch refinement**: i bin armonici femminili confermati da pYIN vengono elevati a ≥ 0.85; i bin armonici maschili vengono soppressi a ≤ 0.08.

#### 5. Reconstruction (DSP)
La maschera IRM viene applicata allo STFT complesso (preservando la fase originale), poi si applica la **ISTFT** per tornare al dominio temporale. Un passo finale di **speech enhancement** (`noisereduce` + peak normalization) rifinisce il segnale ricostruito.

---

## Riferimenti teorici

- **Cherry, E. C.** (1953). *Some Experiments on the Recognition of Speech, with One and with Two Ears*. The Journal of the Acoustical Society of America, 25(5), 975–979. — Definizione originale del *cocktail party problem*.
- **Bregman, A. S.** (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound*. MIT Press. — Fondamento teorico della percezione uditiva e della segregazione delle sorgenti.
- **Wang, D., & Brown, G. J.** (2006). *Computational Auditory Scene Analysis: Principles, Algorithms, and Applications*. Wiley-IEEE Press.
- **Hyvärinen, A., & Oja, E.** (2000). *Independent Component Analysis: Algorithms and Applications*. Neural Networks, 13(4-5), 411–430.

---

## Licenza

Da definire.
