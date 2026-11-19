# Audio AI Projects

![CI](https://github.com/ninuxi/audio-ai-projects/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/github/license/ninuxi/audio-ai-projects)

> 🇬🇧 [English version](README.md)

Un percorso in sei moduli di analisi audio e machine learning in Python, costruito passo dopo passo nel 2025: dall'estrazione di feature e dalla visualizzazione fino a un piccolo servizio API per catalogare archivi sonori.

Sono un fonico con oltre 20 anni di lavoro tra live e installazioni e mi sto spostando verso l'AI engineering. In questo repository ho imparato e messo in pratica la parte Python: elaborazione del segnale con librosa, machine learning classico con scikit-learn, container con Docker, un'API REST, test e integrazione continua.

**Stato:** progetto personale di portfolio e di studio. Non è in uso presso nessuna istituzione. Gli archivi e i musei citati nella documentazione dei moduli (RAI Teche, MAXXI) sono scenari usati per impostare gli esercizi, non clienti.

## Moduli

| Modulo | Cosa fa | Strumenti principali |
| --- | --- | --- |
| [week1-audio-visualizer](01-fundamentals/week1-audio-visualizer) | Carica audio, disegna forma d'onda, spettro e spettrogramma, estrae MFCC e feature spettrali | librosa, NumPy, Matplotlib |
| [week2-batch-processor](01-fundamentals/week2-batch-processor) | Elabora cartelle di file audio: validazione, conversione di formato, normalizzazione, riduzione del rumore, estrazione di metadati, report di qualità | librosa, soundfile, mutagen, pandas |
| [week3-call-analytics](01-fundamentals/week3-call-analytics) | Rilevamento della voce e riconoscimento del parlante su registrazioni di chiamate, a partire da feature MFCC | scikit-learn (Random Forest), librosa |
| [week4-cultural-ai](01-fundamentals/week4-cultural-ai) | Classificazione di genere e contenuto per audio d'archivio, con generazione di metadati da descrizioni testuali | scikit-learn (SVM, TF-IDF), librosa |
| [week5-production](01-fundamentals/week5-production) | Racchiude l'analisi in un servizio con configurazione, log, health check e immagine Docker | Docker, docker-compose, SQLite, Redis opzionale |
| [week6-production-systems](01-fundamentals/week6-production-systems) | Prototipo per la digitalizzazione di archivi: valutazione della qualità, rilevamento del degrado, catalogazione automatica, API REST con autenticazione | FastAPI, SQLite, scikit-learn |

Le demo di week3 e week4 si addestrano e girano su segnali sintetici, e la demo di week4 stampa una classificazione simulata, perché non è stato usato un dataset etichettato. Le loro cifre di accuratezza non sono verificate.

## Avvio rapido

```bash
git clone https://github.com/ninuxi/audio-ai-projects.git
cd audio-ai-projects

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt pytest
python -m pytest tests          # 25 test
```

Ogni modulo ha il suo README e, dove serve, il suo `requirements.txt`.

## Prossimi passi

- Sostituire i dati sintetici di week3 e week4 con un dataset pubblico etichettato e misurare l'accuratezza reale.
- Provare embedding audio preaddestrati (per esempio PANNs o CLAP) al posto delle feature MFCC fatte a mano.
- Spostare l'API di week6 su un database vero e aggiungere test di integrazione.

## Altri progetti

- **MOOD**: analisi video e audio in tempo reale che pilota suono, luci e video nelle installazioni (Python, OpenCV, Node.js, OSC).
- **auto-eq-agente**: backend FastAPI con agenti LLM che leggono file di misura degli altoparlanti e propongono l'equalizzazione.

## Contatti

Antonio Mainenti · Roma · oggettosonoro@gmail.com · [LinkedIn](https://www.linkedin.com/in/mainenti/) · [mainenti.net](https://www.mainenti.net/)

## Licenza

MIT, vedi [LICENSE](LICENSE).
