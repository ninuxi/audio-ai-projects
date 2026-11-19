# Audio AI Projects

![CI](https://github.com/ninuxi/audio-ai-projects/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/github/license/ninuxi/audio-ai-projects)

> 🇮🇹 [Versione italiana](README_IT.md)

A six-module path in Python audio analysis and machine learning, built step by step in 2025: from feature extraction and visualization to a small API service for cataloguing sound archives.

I am a sound engineer with 20+ years of live and installation work, moving into AI engineering. This repository is where I learned and practised the Python side: signal processing with librosa, classic machine learning with scikit-learn, packaging with Docker, a REST API, tests and CI.

**Status:** personal portfolio and learning project. It is not deployed at any institution. Archives and museums mentioned in the module docs (RAI Teche, MAXXI) are design scenarios used to frame the exercises, not clients.

## Modules

| Module | What it does | Main tools |
| --- | --- | --- |
| [week1-audio-visualizer](01-fundamentals/week1-audio-visualizer) | Loads audio, plots waveform, spectrum and spectrogram, extracts MFCC and spectral features | librosa, NumPy, Matplotlib |
| [week2-batch-processor](01-fundamentals/week2-batch-processor) | Processes folders of audio files: validation, format conversion, normalization, noise reduction, metadata extraction, quality reports | librosa, soundfile, mutagen, pandas |
| [week3-call-analytics](01-fundamentals/week3-call-analytics) | Voice activity detection and speaker recognition on call recordings, from MFCC features | scikit-learn (Random Forest), librosa |
| [week4-cultural-ai](01-fundamentals/week4-cultural-ai) | Genre and content classification for archive audio, plus metadata generation from text descriptions | scikit-learn (SVM, TF-IDF), librosa |
| [week5-production](01-fundamentals/week5-production) | Wraps the analysis in a service with configuration, logging, health checks and a Docker image | Docker, docker-compose, SQLite, optional Redis |
| [week6-production-systems](01-fundamentals/week6-production-systems) | Archive digitization prototype: quality assessment, degradation detection, automatic cataloguing, REST API with authentication | FastAPI, SQLite, scikit-learn |

The week3 and week4 demos train and run on synthetic signals, and the week4 demo prints a simulated classification, because no labelled dataset was used. Treat their accuracy figures as untested.

## Quick start

```bash
git clone https://github.com/ninuxi/audio-ai-projects.git
cd audio-ai-projects

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt pytest
python -m pytest tests          # 25 tests
```

Each module has its own README and, where needed, its own `requirements.txt`.

## What I would do next

- Replace the synthetic data in week3 and week4 with a public labelled dataset and report real accuracy.
- Try pretrained audio embeddings (for example PANNs or CLAP) instead of hand-made MFCC features.
- Move the week6 API to a proper database and add integration tests.

## Related work

- **MOOD**: real-time computer vision and audio analysis that drives sound, light and video in installations (Python, OpenCV, Node.js, OSC).
- **auto-eq-agente**: FastAPI backend with LLM agents that parse loudspeaker measurement files and suggest EQ.

## Contact

Antonio Mainenti · Rome · oggettosonoro@gmail.com · [LinkedIn](https://www.linkedin.com/in/mainenti/) · [mainenti.net](https://www.mainenti.net/)

## License

MIT, see [LICENSE](LICENSE).
