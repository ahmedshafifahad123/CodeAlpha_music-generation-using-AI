import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, converter, instrument, note, stream


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "music_lstm.keras"
VOCAB_PATH = ARTIFACTS_DIR / "vocabulary.pkl"


def read_midi_tokens(data_dir: Path) -> list[str]:
    """Read pitches and chords from every MIDI file in data_dir."""
    midi_files = list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi"))
    if not midi_files:
        raise FileNotFoundError(
            f"No .mid or .midi files found in '{data_dir}'. Add MIDI files first."
        )

    tokens: list[str] = []
    for midi_file in midi_files:
        try:
            score = converter.parse(midi_file)
            parts = instrument.partitionByInstrument(score)
            elements = parts.parts[0].recurse() if parts else score.flat.notes
            for element in elements:
                if isinstance(element, note.Note):
                    tokens.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    tokens.append(".".join(str(p.midi) for p in element.pitches))
            print(f"Read {midi_file.name}")
        except Exception as error:
            print(f"Skipped {midi_file.name}: {error}")

    if not tokens:
        raise ValueError("MIDI files were found, but no notes/chords could be read.")
    return tokens


def make_sequences(tokens: list[str], sequence_length: int):
    vocabulary = sorted(set(tokens))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    encoded = [token_to_id[token] for token in tokens]

    inputs, targets = [], []
    for start in range(len(encoded) - sequence_length):
        inputs.append(encoded[start : start + sequence_length])
        targets.append(encoded[start + sequence_length])
    if len(inputs) < 2:
        raise ValueError(
            f"Not enough notes. Use more MIDI data or set --sequence-length below {sequence_length}."
        )
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32), vocabulary


def build_model(vocabulary_size: int, sequence_length: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length,)),
            tf.keras.layers.Embedding(vocabulary_size, 128),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(vocabulary_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args):
    tf.keras.utils.set_random_seed(args.seed)
    tokens = read_midi_tokens(Path(args.data))
    x, y, vocabulary = make_sequences(tokens, args.sequence_length)
    print(f"\nNotes/chords: {len(tokens)} | vocabulary: {len(vocabulary)} | training sequences: {len(x)}")

    model = build_model(len(vocabulary), args.sequence_length)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="loss"),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
    ]
    model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    model.save(MODEL_PATH)
    with VOCAB_PATH.open("wb") as file:
        pickle.dump({"vocabulary": vocabulary, "sequence_length": args.sequence_length}, file)
    print(f"\nDone. Saved model to: {MODEL_PATH}")


def token_to_element(token: str):
    if "." in token or token.isdigit():
        return chord.Chord([int(value) for value in token.split(".")])
    return note.Note(token)


def generate(args):
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        raise FileNotFoundError("Model not found. Run the train command first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with VOCAB_PATH.open("rb") as file:
        saved = pickle.load(file)
    vocabulary = saved["vocabulary"]
    sequence_length = saved["sequence_length"]
    id_to_token = dict(enumerate(vocabulary))

    random.seed(args.seed)
    pattern = [random.randrange(len(vocabulary)) for _ in range(sequence_length)]
    generated = []
    for _ in range(args.num_notes):
        probabilities = model.predict(np.array([pattern]), verbose=0)[0]
        probabilities = np.log(probabilities + 1e-9) / args.temperature
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        next_id = np.random.choice(len(vocabulary), p=probabilities)
        generated.append(id_to_token[next_id])
        pattern = pattern[1:] + [next_id]

    output = stream.Stream()
    offset = 0.0
    for token in generated:
        element = token_to_element(token)
        element.offset = offset
        element.quarterLength = 0.5
        output.append(element)
        offset += 0.5
    output_path = PROJECT_DIR / args.output
    output.write("midi", fp=output_path)
    print(f"Generated {len(generated)} notes/chords: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MIDI music generation with an LSTM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on MIDI files")
    train_parser.add_argument("--data", default="data", help="Folder containing .mid/.midi files")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--sequence-length", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=42)

    generate_parser = subparsers.add_parser("generate", help="Create a MIDI file")
    generate_parser.add_argument("--num-notes", type=int, default=300)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--output", default="generated_music.mid")
    generate_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args) if args.command == "train" else generate(args)


if __name__ == "__main__":
    main()
