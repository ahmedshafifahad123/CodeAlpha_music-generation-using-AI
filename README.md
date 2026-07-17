# Music Generation with AI

A beginner-friendly deep learning project that learns patterns from MIDI files and creates new MIDI music. The model uses an LSTM (Long Short-Term Memory) neural network built with TensorFlow.

## What this project does

1. Collects MIDI music files such as classical or jazz.
2. Reads notes and chords from those files.
3. Converts them into sequences that a neural network can learn from.
4. Trains an LSTM model on the sequences.
5. Generates a new sequence of notes/chords.
6. Saves the generated sequence as a `.mid` file that can be played in MuseScore Studio.

## Features

- Reads both `.mid` and `.midi` files, including files inside subfolders.
- Learns individual notes and chords.
- Saves the trained model and vocabulary for later use.
- Generates MIDI music with adjustable length and creativity.
- Runs on CPU in Windows; a GPU is not required.

## Technologies used

- Python 3.11
- TensorFlow / Keras
- NumPy
- music21
- MIDI files

## Project structure

```text
Music generation with AI/
├── data/                         # Put training MIDI files here
├── artifacts/                    # Created after training (ignored by Git)
│   ├── music_lstm.keras
│   └── vocabulary.pkl
├── screenshots/                  # Add your project screenshots here
├── music.py                      # Main program
├── requirements.txt              # Python dependencies
└── README.md
```

## Prerequisites

Install the following before starting:

- [Python 3.11 (64-bit)](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [MuseScore Studio](https://musescore.org/en/download) to listen to the output MIDI files. MuseScore Studio is free; choose the option **without MuseHub** if you do not want the MuseHub installer.

> TensorFlow is not currently compatible with Python 3.14 in this project. Use Python 3.11.

## Installation

Open this project folder in VS Code. Then open a PowerShell terminal with **Terminal → New Terminal**.

### 1. Create the virtual environment

```powershell
cd "E:\New folder\Music generation with AI"
py -3.11 -m venv .venv
```

### 2. Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of the terminal line.

### 3. Install the required packages

```powershell
python -m pip install --upgrade pip
pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt
```

### 4. Check the installation

```powershell
python -c "import numpy, music21, tensorflow; print('Packages installed correctly')"
```

## Add the training data

Create or use the `data` folder in the project. Add MIDI files with the extension `.mid` or `.midi` to it.

```text
data/
├── piano_piece_1.mid
├── piano_piece_2.mid
└── more_music.midi
```

Use music from one style for better results—for example, piano classical MIDI files. Start with at least 20–50 files; more varied, legally obtained data generally improves the output.

## Train the model

For a quick test, train for 5 epochs:

```powershell
python music.py train --data data --epochs 5 --sequence-length 50
```

For a more complete training run:

```powershell
python music.py train --data data --epochs 50
```

Training creates these files in `artifacts/`:

- `music_lstm.keras` — the trained LSTM model.
- `vocabulary.pkl` — the notes/chords used by the model.

## Generate music

Generate a standard 300-note MIDI file:

```powershell
python music.py generate --num-notes 300 --output generated_music.mid
```

Generate a different song:

```powershell
python music.py generate --num-notes 300 --output song2.mid --temperature 1.1 --seed 99
```

The generated MIDI file is saved in the project folder. Open it in MuseScore Studio using **File → Open**, then press the ▶ Play button.

### Generation options

| Option | Meaning | Example |
| --- | --- | --- |
| `--num-notes` | Number of notes/chords to generate | `--num-notes 500` |
| `--output` | Name of the generated MIDI file | `--output my_song.mid` |
| `--temperature` | Creativity of the output | `0.7` = safer, `1.2` = more varied |
| `--seed` | Starting random seed | `--seed 123` |

## Screenshots

![Generated MIDI in MuseScore](assets/screenshots/musescore-output.png)



## How the AI works

The program converts each note or chord from the MIDI files into a token. It creates many input sequences, each with a target next note. The LSTM learns the relationship between a sequence of notes and the next likely note. During generation, the trained model predicts one note at a time and writes the result to a new MIDI file.

```text
MIDI dataset → notes/chords → number sequences → LSTM training
→ predicted notes/chords → generated .mid file → MuseScore playback
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'` or `tensorflow`

Activate the virtual environment and install requirements again:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### PowerShell says scripts are disabled

Run this in the current terminal, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### `No .mid or .midi files found`

Add MIDI files to the `data` folder, then run the train command again.

### TensorFlow download stops partway through

This is usually an internet interruption. Retry with:

```powershell
pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt
```

### TensorFlow information/warning messages appear

Messages about `oneDNN`, CPU instructions, or native Windows GPU support are informational. The program can still train and generate MIDI music on your CPU.

## Future improvements

- Use a larger and cleaner MIDI dataset.
- Preserve note duration and timing instead of using a fixed duration.
- Add separate models for classical, jazz, and other genres.
- Train longer or use more advanced Transformer-based music models.
- Build a web interface for generating and downloading songs.

## License

This project is for learning and educational use. Ensure you have permission to use any MIDI files included in your dataset.

## Github
bash
```
https://github.com/ahmedshafifahad123/CodeAlpha_Music-generation-with-AI.git
```
