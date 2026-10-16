#!/usr/bin/env python3
"""Standalone symbolic NRPS classifier using logistic regression.

Dependencies:
    Python 3.10+, NumPy, SciPy

Expected CSV columns:
    Training/evaluation: sequence,produced (optional: id)
    Prediction:          sequence (optional: id)

Examples:
    # Stratified 80/20 split, train, report held-out accuracy, and save model
    python symbolic_nrps_logistic.py train data.csv model.npz

    # Train on one CSV and evaluate on a separate labeled CSV
    python symbolic_nrps_logistic.py train train.csv model.npz --test-csv test.csv

    # Evaluate an existing model
    python symbolic_nrps_logistic.py evaluate test.csv model.npz

    # Predict an unlabeled CSV
    python symbolic_nrps_logistic.py predict sequences.csv model.npz predictions.csv

The feature recipe is:
  * 39 length/composition/quality/known-motif measurements;
  * train-selected hashed six-residue motif log-counts (8,000 by default);
  * repeated 50-residue N- and C-terminal signatures and their pairs.

No row ID is used as a model feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_CHARACTERS = "*X"
KNOWN_MOTIFS = (
    "SSGTT",
    "SSGST",
    "YTSGST",
    "GGHSL",
    "YKTGDL",
    "RTGDL",
    "LPDGN",
    "KAGG",
    "HHIV",
    "HHLI",
    "GPTE",
    "GEL",
)
KMER_LENGTH = 6
TERMINAL_LENGTH = 50
HASH_DIMENSION = 1 << 20
HASH_MASK = np.uint64(HASH_DIMENSION - 1)
HASH_MULTIPLIER = np.uint64(0xBF58476D1CE4E5B9)

CHARACTER_CODES = np.full(256, 22, dtype=np.uint64)
for _index, _character in enumerate(AMINO_ACIDS + SPECIAL_CHARACTERS):
    CHARACTER_CODES[ord(_character)] = _index + 1


def parse_label(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized produced label: {value!r}")


def read_csv(path: Path, require_labels: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence"} | ({"produced"} if require_labels else set())
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    if any(not row["sequence"] for row in rows):
        raise ValueError(f"{path} contains an empty sequence")
    if require_labels:
        for row in rows:
            parse_label(row["produced"])
    return rows


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([parse_label(row["produced"]) for row in rows], dtype=float)


def hashed_kmer_counts(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic hash bins and counts for overlapping 6-mers."""
    encoded = CHARACTER_CODES[
        np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    ]
    count = len(encoded) - KMER_LENGTH + 1
    if count <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    packed = np.zeros(count, dtype=np.uint64)
    for offset in range(KMER_LENGTH):
        packed |= encoded[offset : offset + count] << np.uint64(5 * offset)
    packed ^= packed >> np.uint64(30)
    packed *= HASH_MULTIPLIER
    packed ^= packed >> np.uint64(27)
    hashed = (packed & HASH_MASK).astype(np.int32)
    return np.unique(hashed, return_counts=True)


def handcrafted_features(sequence: str) -> np.ndarray:
    length = len(sequence)
    counts = Counter(sequence)
    valid = sum(counts[aa] for aa in AMINO_ACIDS)
    scaled_length = length / 10_000.0
    values = [
        math.log(length),
        scaled_length,
        scaled_length**2,
        sequence.count("GG") / 10.0,
        counts["*"],
        counts["X"],
        (length - valid) / length,
    ]
    values.extend(counts[aa] / length for aa in AMINO_ACIDS)
    values.extend(sequence.count(motif) / 10.0 for motif in KNOWN_MOTIFS)
    return np.asarray(values, dtype=float)


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "n": len(truth),
        "prevalence": float(np.mean(truth)),
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "predicted_positive_rate": float(np.mean(prediction)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def stratified_split(
    rows: list[dict[str, str]], test_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    grouped: dict[bool, list[int]] = {False: [], True: []}
    for index, row in enumerate(rows):
        grouped[parse_label(row["produced"])].append(index)
    if not all(grouped.values()):
        raise ValueError("Both produced classes are required")

    rng = random.Random(seed)
    test_indices: set[int] = set()
    for indices in grouped.values():
        rng.shuffle(indices)
        test_size = max(1, round(test_fraction * len(indices)))
        test_indices.update(indices[:test_size])
    train = [row for index, row in enumerate(rows) if index not in test_indices]
    test = [row for index, row in enumerate(rows) if index in test_indices]
    return train, test


class SymbolicLogisticModel:
    def __init__(
        self,
        selected_bins: np.ndarray,
        hand_mean: np.ndarray,
        hand_scale: np.ndarray,
        prefix_values: np.ndarray,
        suffix_values: np.ndarray,
        pair_values: np.ndarray,
        weights: np.ndarray,
        threshold: float = 0.0,
    ) -> None:
        self.selected_bins = selected_bins
        self.hand_mean = hand_mean
        self.hand_scale = hand_scale
        self.prefix_values = prefix_values
        self.suffix_values = suffix_values
        self.pair_values = pair_values
        self.weights = weights
        self.threshold = threshold

    @staticmethod
    def repeated_values(values: list[str], width: int) -> np.ndarray:
        counts = Counter(values)
        kept = sorted(value for value, count in counts.items() if count >= 2)
        return np.asarray(kept, dtype=f"<U{width}")

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, str]],
        number_of_kmers: int,
        minimum_document_frequency: int,
        l2: float,
        maximum_iterations: int,
        threshold: float,
    ) -> tuple["SymbolicLogisticModel", dict[str, object]]:
        sequences = [row["sequence"] for row in rows]
        y = labels(rows)
        if len(np.unique(y)) != 2:
            raise ValueError("Training data must contain both classes")

        documents = [hashed_kmer_counts(sequence) for sequence in sequences]
        positive_df = np.zeros(HASH_DIMENSION, dtype=np.int16)
        negative_df = np.zeros(HASH_DIMENSION, dtype=np.int16)
        for (bins, _), label in zip(documents, y, strict=True):
            (positive_df if label else negative_df)[bins] += 1

        total = positive_df.astype(np.int32) + negative_df
        difference = positive_df.astype(np.int32) - negative_df
        ranking = np.where(
            total >= minimum_document_frequency,
            difference * difference / (total + 10.0),
            -1.0,
        )
        eligible = int(np.sum(ranking >= 0))
        if eligible < number_of_kmers:
            raise ValueError(
                f"Requested {number_of_kmers} motif bins, but only {eligible} "
                "meet the minimum document frequency. Reduce --n-kmers or --min-df."
            )
        selected = np.argpartition(ranking, -number_of_kmers)[-number_of_kmers:]
        selected = selected[np.argsort(ranking[selected])[::-1]].astype(np.int32)

        hand = np.vstack([handcrafted_features(sequence) for sequence in sequences])
        hand_mean = hand.mean(axis=0)
        hand_scale = hand.std(axis=0)
        hand_scale[hand_scale < 1e-10] = 1.0

        prefixes = cls.repeated_values(
            [sequence[:TERMINAL_LENGTH] for sequence in sequences], TERMINAL_LENGTH
        )
        suffixes = cls.repeated_values(
            [sequence[-TERMINAL_LENGTH:] for sequence in sequences], TERMINAL_LENGTH
        )
        pairs = cls.repeated_values(
            [
                sequence[:TERMINAL_LENGTH]
                + "|"
                + sequence[-TERMINAL_LENGTH:]
                for sequence in sequences
            ],
            2 * TERMINAL_LENGTH + 1,
        )

        model = cls(
            selected,
            hand_mean,
            hand_scale,
            prefixes,
            suffixes,
            pairs,
            np.empty(0),
            threshold,
        )
        matrix = model.feature_matrix(sequences, documents)

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            scores = matrix @ weights
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
            loss = (
                np.logaddexp(0, scores).sum()
                - y @ scores
                + l2 / 2.0 * weights[1:].dot(weights[1:])
            )
            gradient = np.asarray(matrix.T @ (probabilities - y)).ravel()
            gradient[1:] += l2 * weights[1:]
            return float(loss), gradient

        optimization = minimize(
            objective,
            np.zeros(matrix.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maximum_iterations, "ftol": 1e-8, "maxls": 30},
        )
        model.weights = optimization.x
        train_metrics = classification_metrics(y, model.predict(sequences))
        train_metrics["optimization"] = {
            "iterations": int(optimization.nit),
            "success": bool(optimization.success),
            "message": str(optimization.message),
        }
        return model, train_metrics

    @staticmethod
    def category_matrix(
        observed: list[str], known: np.ndarray
    ) -> sparse.csr_matrix:
        lookup = {value: index for index, value in enumerate(known.tolist())}
        row_indices: list[int] = []
        column_indices: list[int] = []
        for row_index, value in enumerate(observed):
            column_index = lookup.get(value)
            if column_index is not None:
                row_indices.append(row_index)
                column_indices.append(column_index)
        return sparse.csr_matrix(
            (np.ones(len(row_indices)), (row_indices, column_indices)),
            shape=(len(observed), len(known)),
        )

    def feature_matrix(
        self,
        sequences: list[str],
        documents: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> sparse.csr_matrix:
        selected_map = np.full(HASH_DIMENSION, -1, dtype=np.int32)
        selected_map[self.selected_bins] = np.arange(len(self.selected_bins))
        if documents is None:
            documents = [hashed_kmer_counts(sequence) for sequence in sequences]

        indptr = [0]
        index_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        for bins, counts in documents:
            indices = selected_map[bins]
            keep = indices >= 0
            index_chunks.append(indices[keep])
            value_chunks.append(np.log1p(counts[keep].astype(float)))
            indptr.append(indptr[-1] + int(np.sum(keep)))
        motif_indices = np.concatenate(index_chunks)
        motif_values = np.concatenate(value_chunks)
        motif_matrix = sparse.csr_matrix(
            (motif_values, motif_indices, np.asarray(indptr)),
            shape=(len(sequences), len(self.selected_bins)),
        )

        hand = np.vstack([handcrafted_features(sequence) for sequence in sequences])
        hand_matrix = sparse.csr_matrix((hand - self.hand_mean) / self.hand_scale)
        prefix_matrix = self.category_matrix(
            [sequence[:TERMINAL_LENGTH] for sequence in sequences],
            self.prefix_values,
        )
        suffix_matrix = self.category_matrix(
            [sequence[-TERMINAL_LENGTH:] for sequence in sequences],
            self.suffix_values,
        )
        pair_matrix = self.category_matrix(
            [
                sequence[:TERMINAL_LENGTH]
                + "|"
                + sequence[-TERMINAL_LENGTH:]
                for sequence in sequences
            ],
            self.pair_values,
        )
        return sparse.hstack(
            [
                np.ones((len(sequences), 1)),
                hand_matrix,
                motif_matrix,
                prefix_matrix,
                suffix_matrix,
                pair_matrix,
            ],
            format="csr",
        )

    @property
    def feature_counts(self) -> dict[str, int]:
        return {
            "intercept": 1,
            "handcrafted": len(self.hand_mean),
            "selected_6mer_bins": len(self.selected_bins),
            "n_terminal_signatures": len(self.prefix_values),
            "c_terminal_signatures": len(self.suffix_values),
            "terminal_pairs": len(self.pair_values),
            "total": len(self.weights),
        }

    def decision_function(self, sequences: list[str]) -> np.ndarray:
        return np.asarray(self.feature_matrix(sequences) @ self.weights).ravel()

    def predict(self, sequences: list[str]) -> np.ndarray:
        return self.decision_function(sequences) >= self.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            selected_bins=self.selected_bins,
            hand_mean=self.hand_mean,
            hand_scale=self.hand_scale,
            prefix_values=self.prefix_values,
            suffix_values=self.suffix_values,
            pair_values=self.pair_values,
            weights=self.weights,
            threshold=np.asarray([self.threshold]),
        )

    @classmethod
    def load(cls, path: Path) -> "SymbolicLogisticModel":
        with np.load(path, allow_pickle=False) as checkpoint:
            return cls(
                checkpoint["selected_bins"],
                checkpoint["hand_mean"],
                checkpoint["hand_scale"],
                checkpoint["prefix_values"],
                checkpoint["suffix_values"],
                checkpoint["pair_values"],
                checkpoint["weights"],
                float(checkpoint["threshold"][0]),
            )


def train_command(args: argparse.Namespace) -> int:
    all_rows = read_csv(args.csv, require_labels=True)
    if args.test_csv is None:
        train_rows, test_rows = stratified_split(
            all_rows, args.test_fraction, args.seed
        )
        split_description = {
            "method": "stratified random split",
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        }
    else:
        train_rows = all_rows
        test_rows = read_csv(args.test_csv, require_labels=True)
        split_description = {"method": "separate test CSV", "path": str(args.test_csv)}

    model, train_metrics = SymbolicLogisticModel.fit(
        train_rows,
        number_of_kmers=args.n_kmers,
        minimum_document_frequency=args.min_df,
        l2=args.l2,
        maximum_iterations=args.max_iterations,
        threshold=args.threshold,
    )
    model.save(args.model)
    test_truth = labels(test_rows)
    test_predictions = model.predict([row["sequence"] for row in test_rows])
    output = {
        "model_path": str(args.model),
        "split": split_description,
        "train_label_counts": Counter(row["produced"] for row in train_rows),
        "test_label_counts": Counter(row["produced"] for row in test_rows),
        "feature_counts": model.feature_counts,
        "train_metrics": train_metrics,
        "test_metrics": classification_metrics(test_truth, test_predictions),
    }
    print(json.dumps(output, indent=2))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    rows = read_csv(args.csv, require_labels=True)
    model = SymbolicLogisticModel.load(args.model)
    truth = labels(rows)
    predictions = model.predict([row["sequence"] for row in rows])
    print(
        json.dumps(
            {
                "model_path": str(args.model),
                "csv": str(args.csv),
                "metrics": classification_metrics(truth, predictions),
            },
            indent=2,
        )
    )
    return 0


def predict_command(args: argparse.Namespace) -> int:
    rows = read_csv(args.csv, require_labels=False)
    model = SymbolicLogisticModel.load(args.model)
    predictions = model.predict([row["sequence"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "produced"], lineterminator="\n"
        )
        writer.writeheader()
        for index, (row, prediction) in enumerate(
            zip(rows, predictions, strict=True)
        ):
            writer.writerow(
                {
                    "id": row.get("id") or f"row_{index:06d}",
                    "produced": "True" if prediction else "False",
                }
            )
    print(f"Wrote {len(rows)} predictions to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="fit and save a model")
    train_parser.add_argument("csv", type=Path, help="labeled training CSV")
    train_parser.add_argument("model", type=Path, help="output .npz checkpoint")
    train_parser.add_argument("--test-csv", type=Path)
    train_parser.add_argument("--test-fraction", type=float, default=0.20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--n-kmers", type=int, default=8_000)
    train_parser.add_argument("--min-df", type=int, default=5)
    train_parser.add_argument("--l2", type=float, default=1.0)
    train_parser.add_argument("--max-iterations", type=int, default=180)
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="logistic decision threshold; zero is appropriate when training on natural prevalence",
    )
    train_parser.set_defaults(function=train_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a saved model on a labeled CSV"
    )
    evaluate_parser.add_argument("csv", type=Path)
    evaluate_parser.add_argument("model", type=Path)
    evaluate_parser.set_defaults(function=evaluate_command)

    predict_parser = subparsers.add_parser(
        "predict", help="predict an unlabeled CSV"
    )
    predict_parser.add_argument("csv", type=Path)
    predict_parser.add_argument("model", type=Path)
    predict_parser.add_argument("output", type=Path)
    predict_parser.set_defaults(function=predict_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
