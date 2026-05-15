# ROADMAP — Piano d'azione di implementazione

> **Stato:** in corso — Fasi 0–6 completate (pipeline funzionante end-to-end). Mancano notebook di valutazione e test automatici con metriche oggettive.
> **Ultimo aggiornamento:** 2026-05-15.

Documento che traccia il piano d'azione completo per l'implementazione del sistema. Da consultare e aggiornare a ogni inizio fase. Le decisioni prese vanno spostate dalla sezione "Punti aperti" al corpo del documento.

---

## Struttura in 6 fasi incrementali

Ogni fase produce qualcosa di funzionante e testabile. Le fasi rispettano le dipendenze tecniche: il DSP fornisce feature alla AI, la AI restituisce decisioni al DSP per la separazione.

---

### Fase 0 — Fondazioni (infrastruttura)

**Obiettivo:** basi I/O e utility pronte prima di scrivere logica di dominio.

1. `src/utils.py` — Funzioni di I/O audio (load/save WAV con `librosa`/`soundfile`), resampling a frequenza standard (16 kHz), logging strutturato, validazione input.
2. `tests/test_utils.py` — Test su round-trip load→save, conversioni di sample rate.

**Deliverable:** poter caricare/salvare un file audio in modo affidabile da qualsiasi modulo.

---

### Fase 1 — DSP: trasformazione e feature

**Obiettivo:** dato un segnale audio, produrre un vettore di feature per ogni finestra temporale.

1. `src/dsp/stft.py` — Wrapper su `librosa.stft` / `istft` con parametri standardizzati (n_fft, hop_length, window). Mantenere centralizzata la configurazione perché STFT e ISTFT devono usare gli stessi parametri.
2. `src/dsp/features.py` — Estrattori per:
   - MFCC (timbro)
   - Pitch via pYIN (frequenza fondamentale)
   - RMS energy (intensità)
   - Spectral centroid + rolloff (colore timbrico)
   - `extract_all(audio) → feature_matrix` che restituisce tutto concatenato per frame.
3. `tests/test_features.py` — Sanity check su segnali sintetici (sinusoidi a frequenza nota → pitch corretto, silenzio → energia ~0).

**Deliverable:** pipeline DSP fino al vettore di feature, testata su segnali noti.

---

### Fase 2 — Dataset e ground truth

**Obiettivo:** dati su cui addestrare e valutare. **Fase critica** — senza dati non si va da nessuna parte.

Due opzioni in discussione:

- **Opzione A — Dataset pubblico:** subset di **LibriSpeech** (voci pulite ed etichettate per speaker) + mix sintetici controllati (es. 2 voci sommate a SNR noto).
- **Opzione B — Dataset minimale custom:** 5-10 clip per voce maschile e 5-10 per voce femminile, generare mix.

**Piano operativo:**
1. `src/utils.py` — `make_mixture(voice_a, voice_b, snr_db)` per creare mix sintetici riproducibili.
2. `notebooks/01_data_exploration.ipynb` — Esplorare i dati, ascoltare i mix, visualizzare spettrogrammi.

**Deliverable:** piccolo dataset `data/raw/` con mix etichettati.

---

### Fase 3 — AI: classificatore di attenzione

**Obiettivo:** dato un vettore di feature, decidere quale parlante è il target.

**Approccio iniziale (semplice e interpretabile):**

1. `src/ai/train.py` — Script che:
   - Carica audio etichettati per speaker target,
   - Estrae feature via `src/dsp/features`,
   - Addestra un classificatore (start: **Random Forest** — robusto, poco tuning, gestisce feature eterogenee),
   - Salva il modello via `joblib` in `models/`.
2. `src/ai/classifier.py` — Classe `SpeakerClassifier` che incapsula load/predict.
3. `src/ai/attention.py` — Logica di alto livello: dato uno spettrogramma multi-speaker, restituisce per ogni frame la probabilità di appartenere al target.
4. `tests/test_classifier.py` — Test su feature note → predizione attesa.

**Deliverable:** classificatore addestrato che, dato un frame audio, decide se è "target" o "interferente".

---

### Fase 4 — DSP: separazione e ricostruzione

**Obiettivo:** dato il mix e le decisioni della AI, ricostruire l'audio del solo target.

1. `src/dsp/separation.py` — **Maschera tempo-frequenza** binaria o soft (Ratio Mask) basata sull'output dell'attention module.
2. Applicazione della maschera allo spettrogramma del mix.
3. Ricostruzione via ISTFT.
4. `tests/test_separation.py` — Test su mix sintetici: il segnale ricostruito deve essere percettivamente più vicino al target che al mix originale (SDR / SI-SDR).

**Deliverable:** audio isolato del parlante target.

---

### Fase 5 — Enhancement post-processing

**Obiettivo:** rifinire l'audio ricostruito riducendo artefatti e residui dell'interferente.

1. `src/dsp/enhancement.py` — Wrapper su `noisereduce` + eventuale filtraggio Wiener leggero su `scipy.signal`.
2. Confronto A/B: audio prima vs. dopo enhancement.

**Deliverable:** versione finale dell'audio isolato, pulita.

---

### Fase 6 — Pipeline end-to-end e valutazione

**Obiettivo:** mettere tutto insieme dietro un'interfaccia CLI unica.

1. `src/pipeline.py` — CLI con `argparse`:
   ```
   python -m src.pipeline --input mix.wav --model models/classifier.joblib --output out.wav
   ```
   Orchestra: load → STFT → features → AI decision → mask → ISTFT → enhancement → save.
2. `notebooks/02_evaluation.ipynb` — Valutazione quantitativa:
   - Metriche: **SI-SDR**, **PESQ**, **STOI**.
   - Confronto con baseline.
   - Visualizzazione spettrogrammi prima/dopo.
3. `tests/test_pipeline.py` — Smoke test end-to-end.

**Deliverable:** sistema completo + report di valutazione con numeri concreti.

---

## Dipendenze tra fasi

```
Fase 0 ──┐
         ├──► Fase 1 (DSP features) ──┐
         │                            ├──► Fase 3 (AI) ──┐
         └──► Fase 2 (Dataset) ───────┘                  │
                                                         ├──► Fase 4 ──► Fase 5 ──► Fase 6
                                      Fase 1 ────────────┘
```

Le Fasi 1 e 2 possono procedere in parallelo dopo la Fase 0. La Fase 3 richiede entrambe.

---

## Stima dello sforzo (relativa)

| Fase | Complessità | Note |
|---|---|---|
| 0 | Bassa | I/O standard |
| 1 | Media | Calibrazione parametri STFT/feature |
| 2 | Alta | La parte più delicata: scelta dati e criterio target |
| 3 | Media | Random Forest è low-effort, ma serve buon training set |
| 4 | Media | Masking è std, ma serve gestire artefatti |
| 5 | Bassa | Post-processing, librerie esistenti |
| 6 | Media | Integrazione + metriche |

---

## Punti aperti — DA DECIDERE prima di iniziare l'implementazione

1. **Dataset:** ✅ LibriSpeech `dev-clean` (~337 MB). Da scaricare su `openslr.org/12/` e inserire in `data/raw/librispeech/dev-clean/`.
2. **Criterio di attenzione:** ✅ Genere vocale (M/F). Partire semplice, alzare l'asticella dopo che il sistema funziona.
3. **Numero di parlanti nel mix:** ✅ 2 voci fisse per v1 (1M + 1F). In futuro il classificatore potrà scalare a N voci in ingresso.
4. **Ordine di lavoro:** ✅ Fase 0+1 avviate subito in parallelo alla preparazione dati.

---

## Stato di avanzamento

| Fase | Stato | Note |
|---|---|---|
| 0 | ✅ Completata | `src/utils.py` + `tests/test_utils.py` |
| 1 | ✅ Completata | `src/dsp/stft.py` + `src/dsp/features.py` (44 feature) + `tests/test_features.py` |
| 2 | ✅ Completata | `src/dsp/dataset.py` — parsing SPEAKERS.TXT, mix M+F + IBM dataset multi-SNR |
| 3 | ✅ Completata | `src/ai/classifier.py`, `train.py`, `attention.py` — CV su 400 sample × 3 SNR |
| 4 | ✅ Completata | `src/dsp/nmf_separation.py` (modulo primario) + `separation.py` (utility/fallback) |
| 5 | ✅ Completata | `src/dsp/enhancement.py` — noisereduce + peak normalization |
| 6 | 🟡 In corso | `src/pipeline.py` + `demo.py` funzionanti; mancano test automatici e notebook valutazione |

> Legenda: ⬜ Non iniziata · 🟡 In corso · ✅ Completata · 🔴 Bug attivo

---

## Miglioramenti implementati (v2 pipeline)

Dopo il primo test manuale (v1 non sopprimeva quasi nulla la voce maschile), sono stati applicati 3 step di miglioramento:

| Step | Modifica | File |
|---|---|---|
| 1 — IBM training | Riaddestramento su frame reali di mix con etichette IBM frame-level. Risolve il domain mismatch: il modello ora vede a training lo stesso tipo di segnale che riceve a inference. | `src/dsp/dataset.py`, `src/ai/train.py` |
| 2 — Pitch mask | Maschera frequenza-selettiva: rileva F0 femminile nel mix (150–310 Hz) e marca i bin armonici come "target". Sopprime la voce maschile nei bin dove non ci sono armoniche femminili. | `src/dsp/separation.py` |
| 3 — Sharpening sigmoid | Sigmoid centrato a 0.5 (`1/(1+exp(-k*(m-0.5)))`): spinge i valori verso 0/1. Corretto rispetto a `mask**power` che schiacciava tutto verso 0. | `src/dsp/separation.py` |

---

## Miglioramenti implementati (v3 pipeline)

Dopo ulteriori test (v2 attenuava entrambe le voci nelle sezioni di sovrapposizione), sono stati applicati i seguenti miglioramenti:

| Step | Modifica | File |
|---|---|---|
| 4 — Feature estese | Da 17 a 44 feature per frame: MFCC delta + delta-delta (26 feature aggiuntive), ZCR, pitch range femminile specifico (150–310 Hz). Migliora la discriminazione M/F nei frame sovrapposti. | `src/dsp/features.py` |
| 5 — Multi-SNR training | Training su mix a -3, 0, +3 dB SNR (invece di solo 0 dB). Il classificatore diventa robusto a diverse proporzioni di energia M/F. | `src/ai/train.py` |
| 6 — Priority masking | Masking a priorità in `separation.py`: UNCERTAIN_CAP=0.25 per frame ambigui, HARMONIC_FLOOR=0.85 per armonici femminili confermati, MALE_SUPPRESSION=0.08 per armonici maschili rilevati. | `src/dsp/separation.py` |
| 7 — NMF separation | Nuovo modulo `nmf_separation.py`: decompone lo spettrogramma in K=8 componenti NMF, assegna ogni componente a F o M via dominant-frame scoring, costruisce IRM per bin. | `src/dsp/nmf_separation.py` |
| 8 — IRM ibrida | L'IRM NMF usa una soft mask lineare `V_f/(V_f+V_m)` (evita il collasso da squaring con sbilanciamento energetico) blended al 65% con le attention weights del classificatore. IRM media: da 0.16 a 0.43+ sui campioni di test. | `src/dsp/nmf_separation.py` |

---

## Prossimi task (in ordine di priorità)

1. **Test automatici** — `tests/test_pipeline.py` (smoke test end-to-end + SI-SDR minimo accettabile)
2. **Notebook di valutazione** — `notebooks/02_evaluation.ipynb` con SI-SDR, PESQ, STOI e spettrogrammi prima/dopo
3. **Fase 6 completa** — chiudere con metriche quantitative
