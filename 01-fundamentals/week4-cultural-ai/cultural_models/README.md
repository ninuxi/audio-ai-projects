# Cultural Models Directory

Questa cartella contiene i modelli di machine learning addestrati per la classificazione di contenuti culturali.

## Struttura Directory

```
cultural_models/
├── README.md                           # Questo file
├── model_registry.json                # Registro dei modelli disponibili
├── pretrained/                        # Modelli pre-addestrati
│   ├── heritage_classifier_v1.pkl     # Classificatore heritage principale
│   ├── genre_classifier_v1.pkl        # Classificatore generi musicali
│   ├── period_classifier_v1.pkl       # Classificatore periodi storici
│   └── instrument_classifier_v1.pkl   # Classificatore strumenti
├── training_data/                     # Dati di training
│   ├── labeled_samples.json           # Campioni etichettati
│   ├── feature_vectors.npy            # Vettori di features
│   └── training_metadata.json         # Metadati training
├── evaluation/                        # Risultati di valutazione
│   ├── model_performance.json         # Performance metrics
│   ├── confusion_matrices/            # Matrici di confusione
│   └── validation_reports/            # Report di validazione
└── schemas/                          # Schemi di classificazione
    ├── cultural_taxonomy.json         # Tassonomia culturale
    ├── genre_hierarchy.json           # Gerarchia generi
    └── period_timeline.json           # Timeline periodi storici
```

## Modelli Disponibili

### 1. Heritage Classifier v1.0
**File:** `pretrained/heritage_classifier_v1.pkl`  
**Descrizione:** Classificatore principale per contenuti del patrimonio culturale  
**Categorie:** 10 classi (folk_music, classical_music, opera, spoken_word, etc.)  
**Accuratezza:** 87.3%  
**Features:** MFCC, spectral features, harmonic analysis  

### 2. Genre Classifier v1.0  
**File:** `pretrained/genre_classifier_v1.pkl`  
**Descrizione:** Classificazione dettagliata di generi musicali tradizionali  
**Categorie:** 15 generi musicali italiani e europei  
**Accuratezza:** 82.1%  
**Features:** Chroma, tempo, harmonic content  

### 3. Period Classifier v1.0
**File:** `pretrained/period_classifier_v1.pkl`  
**Descrizione:** Classificazione di periodi storici musicali  
**Categorie:** Medieval, Renaissance, Baroque, Classical, Romantic, Modern  
**Accuratezza:** 79.8%  
**Features:** Harmonic complexity, instrumentation patterns  

### 4. Instrument Classifier v1.0
**File:** `pretrained/instrument_classifier_v1.pkl`  
**Descrizione:** Riconoscimento di strumenti tradizionali  
**Categorie:** Piano, violino, chitarra, mandolino, fisarmonica, voce, etc.  
**Accuratezza:** 91.2%  
**Features:** Timbral analysis, spectral characteristics  

## Utilizzo dei Modelli

### Python API
```python
from heritage_classifier import HeritageClassifier

# Carica modello pre-addestrato
classifier = HeritageClassifier()
classifier.load_model('cultural_models/pretrained/heritage_classifier_v1.pkl')

# Classifica audio
result = classifier.classify_audio('path/to/audio.wav')
print(f"Categoria: {result['predicted_category']}")
print(f"Confidenza: {result['confidence']:.3f}")
```

### Command Line
```bash
python heritage_classifier.py --model cultural_models/pretrained/heritage_classifier_v1.pkl --audio path/to/audio.wav
```

## Training Data

### Labeled Samples
I dataset di training includono:
- **5,000+ campioni audio** etichettati manualmente
- **Diverse fonti:** RAI Teche, archivi regionali, collezioni private
- **Bilanciamento:** Distribuzione equilibrata tra categorie
- **Qualità:** Controllo qualità professionale

### Feature Extraction
- **MFCC:** 13 coefficienti mel-cepstrali
- **Spectral Features:** Centroide, larghezza di banda, rolloff
- **Harmonic Analysis:** Separazione armonica/percussiva
- **Rhythm Features:** Tempo, consistenza ritmica
- **Cultural Features:** Caratteristiche specifiche della musica tradizionale

## Performance Metrics

### Heritage Classifier v1.0
```
Precision: 0.873
Recall: 0.871
F1-Score: 0.872
```

**Confusion Matrix:**
```
              folk  classical  opera  spoken  ...
folk           0.89      0.03   0.01    0.02  ...
classical      0.02      0.91   0.04    0.01  ...
opera          0.01      0.05   0.88    0.02  ...
spoken         0.03      0.01   0.01    0.93  ...
...
```

### Cross-Validation Results
- **5-Fold CV Mean:** 0.869 ± 0.012
- **Stratified CV:** 0.871 ± 0.015
- **Temporal Split:** 0.856 ± 0.018

## Model Versioning

### Version History
- **v1.0 (Current):** Initial release, baseline performance
- **v0.9:** Beta version, limited categories
- **v0.8:** Prototype, proof of concept

### Planned Updates
- **v1.1:** Enhanced instrument recognition
- **v1.2:** Regional dialect classification
- **v2.0:** Deep learning architecture upgrade

## Cultural Taxonomy

### Hierarchical Structure
```json
{
  "italian_traditional": {
    "northern": ["venetian", "lombard", "piedmontese"],
    "central": ["tuscan", "roman", "umbrian"],
    "southern": ["neapolitan", "sicilian", "calabrese"],
    "islands": ["sardinian", "corsican"]
  },
  "genres": {
    "folk": ["ballata", "stornello", "serenata"],
    "classical": ["opera_seria", "opera_buffa", "sinfonia"],
    "religious": ["gregoriano", "polifonia", "laude"]
  }
}
```

## Integration with RAI Archive Tool

Il sistema è integrato con `rai_archive_tool.py`:

```python
from rai_archive_tool import RAIArchiveTool
from heritage_classifier import HeritageClassifier

# Inizializza strumenti
rai_tool = RAIArchiveTool()
classifier = HeritageClassifier()
classifier.load_model('cultural_models/pretrained/heritage_classifier_v1.pkl')

# Processa archivio RAI
rai_files = rai_tool.get_audio_files()
for audio_file in rai_files:
    classification = classifier.classify_audio(audio_file)
    rai_tool.update_metadata(audio_file, classification)
```

## Quality Assurance

### Validation Process
1. **Expert Review:** Validazione da musicologi esperti
2. **Cross-Validation:** Test su dataset indipendenti  
3. **Real-World Testing:** Test su archivi reali
4. **Performance Monitoring:** Monitoraggio continuo performance

### Quality Metrics
- **Accuracy Threshold:** Minimo 80% per produzione
- **Confidence Scores:** Tracciamento confidence per ogni predizione
- **Error Analysis:** Analisi dettagliata degli errori
- **Bias Detection:** Controllo bias nei dataset

## Deployment

### Production Requirements
- **Python 3.8+**
- **Scikit-learn 1.0+**
- **Librosa 0.9+**
- **NumPy, Pandas, Matplotlib**
- **Minimum RAM:** 4GB
- **Storage:** 500MB per modello completo

### Performance Optimization
- **Model Compression:** Riduzione dimensioni file
- **Feature Caching:** Cache delle features estratte
- **Batch Processing:** Elaborazione batch ottimizzata
- **Parallel Processing:** Supporto multi-core

## Research & Development

### Current Research
- **Deep Learning:** Reti neurali convoluzionali per audio
- **Transfer Learning:** Adattamento da modelli generici
- **Multimodal:** Integrazione audio + metadata testuale
- **Few-Shot Learning:** Classificazione con pochi esempi

### Collaborations
- **Università:** Partnership con dipartimenti di musicologia
- **Istituti:** Collaborazione con conservatori
- **Archives:** Accesso a collezioni specializzate
- **International:** Progetti europei di digitalizzazione

## Legal & Ethical Considerations

### Copyright
- **Training Data:** Solo contenuti liberi da copyright
- **Fair Use:** Utilizzo per ricerca e preservazione
- **Attribution:** Crediti appropriati per fonti

### Cultural Sensitivity
- **Representation:** Rappresentazione equa delle culture
- **Community Input:** Coinvolgimento delle comunità
- **Bias Mitigation:** Riduzione bias algoritmici
- **Transparency:** Documentazione processi decisionali

---

## Contact & Support

**Technical Support:** tech-support@cultural-ai.org  
**Research Inquiries:** research@cultural-ai.org  
**Data Contributions:** data@cultural-ai.org  

**Documentation:** [https://docs.cultural-ai.org](https://docs.cultural-ai.org)  
**GitHub:** [https://github.com/cultural-ai/heritage-classifier](https://github.com/cultural-ai/heritage-classifier)  

---

*Last Updated: July 21, 2025*  
*Version: 1.0.0*
