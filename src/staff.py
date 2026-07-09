"""Step 6 of the pipeline: decide which tracked people are staff, comparing three methods on the labelled tracks.

  rule       feet inside the billing camera's staff_area for most of the track (billing camera only);
  zero-shot  CLIP compares each crop with sentences describing staff and customers, with no training;
  probe      logistic regression trained on frozen CLIP image embeddings of the labelled crops.

A track's score is the mean over its crops. Writes outputs/staff_comparison.md, outputs/staff_scores.csv
and outputs/track_labels.csv, the winning method applied to every track.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from src.config import StoreConfig, load_config
from src.events import drop_short_tracks, group_by_track
from src.geometry import point_in_polygon
from src.track_table import read_tracks_csv

CROPS_INDEX = Path("outputs/crops/index.csv")
EMBEDDINGS_CACHE = Path("outputs/crops/clip_embeddings.npz")
LABELS = Path("data/labels/staff.csv")
# The OpenAI weights were trained with the QuickGELU activation; the "-quickgelu" config matches them.
CLIP_MODEL, CLIP_WEIGHTS = "ViT-B-32-quickgelu", "openai"
STAFF_PROMPTS = [
    "a photo of a shop assistant in a black uniform",
    "a photo of a store employee working behind a counter",
    "a photo of a salesperson restocking shelves",
    "a photo of a beauty advisor at a cosmetics counter",
]
CUSTOMER_PROMPTS = [
    "a photo of a customer shopping in a store",
    "a photo of a shopper browsing cosmetics",
    "a photo of a person walking past a shop window",
    "a photo of a member of the public on the street",
]
OUTER_FOLDS, INNER_FOLDS = 5, 3
IN_STORE_CAMERAS = ("zone_1", "zone_2", "billing")

TrackKey = tuple[str, int]


@dataclass
class Crops:
    tracks: list[TrackKey]
    paths: list[str]


@dataclass
class MethodResult:
    name: str
    scores: dict[TrackKey, float]
    is_staff: dict[TrackKey, bool]


def read_crops(path: Path = CROPS_INDEX) -> Crops:
    with path.open() as file:
        rows = list(csv.DictReader(file))
    return Crops([(row["camera"], int(row["track_id"])) for row in rows], [row["path"] for row in rows])


def read_labels(path: Path = LABELS) -> dict[TrackKey, int]:
    """1 for staff, 0 for not staff. Unsure tracks are left out: they cannot be scored."""
    with path.open() as file:
        return {(row["camera"], int(row["track_id"])): int(row["staff"] == "yes")
                for row in csv.DictReader(file) if row["staff"] in ("yes", "no")}


def read_customer_tracks(path: Path = LABELS) -> set[TrackKey]:
    """Labelled real customers (role "customer"), as opposed to passers-by or false detections."""
    with path.open() as file:
        return {(row["camera"], int(row["track_id"])) for row in csv.DictReader(file) if row["role"] == "customer"}


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(CLIP_MODEL)


def embed_images(paths: list[str], model, preprocess, batch_size: int = 32) -> np.ndarray:
    """One 512-number vector per crop, scaled to length 1 so a dot product is a cosine similarity."""
    if EMBEDDINGS_CACHE.exists():
        cached = np.load(EMBEDDINGS_CACHE, allow_pickle=False)
        if "model" in cached.files and str(cached["model"]) == CLIP_MODEL and list(cached["paths"]) == paths:
            return cached["features"]
    batches = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
            features = model.encode_image(images)
            batches.append((features / features.norm(dim=-1, keepdim=True)).numpy())
    features = np.concatenate(batches)
    np.savez(EMBEDDINGS_CACHE, model=np.array(CLIP_MODEL), paths=np.array(paths), features=features)
    return features


def class_text_vector(prompts: list[str], model, tokenizer) -> np.ndarray:
    """Average of several phrasings of one class (prompt ensembling), so no single wording decides."""
    with torch.no_grad():
        features = model.encode_text(tokenizer(prompts))
    features = features / features.norm(dim=-1, keepdim=True)
    mean = features.mean(dim=0)
    return (mean / mean.norm()).numpy()


def zero_shot_crop_scores(features: np.ndarray, model, tokenizer) -> np.ndarray:
    """Probability that each crop matches the staff sentences rather than the customer sentences."""
    text = np.stack([class_text_vector(CUSTOMER_PROMPTS, model, tokenizer),
                     class_text_vector(STAFF_PROMPTS, model, tokenizer)])
    logits = model.logit_scale.exp().item() * features @ text.T
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def mean_by_track(tracks: list[TrackKey], values: np.ndarray) -> dict[TrackKey, float]:
    totals: dict[TrackKey, list[float]] = {}
    for track, value in zip(tracks, values):
        totals.setdefault(track, []).append(float(value))
    return {track: sum(v) / len(v) for track, v in totals.items()}


def group_ids(tracks: list[TrackKey]) -> np.ndarray:
    """A fixed number per track, so every crop of a track lands in the same fold, the same way every run."""
    number = {track: i for i, track in enumerate(sorted(set(tracks)))}
    return np.array([number[t] for t in tracks])


def fit_probe(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    # class_weight="balanced" makes a missed staff member cost as much in total as a missed customer.
    return LogisticRegression(class_weight="balanced", max_iter=2000).fit(features, labels)


def best_f1_threshold(scores: dict[TrackKey, float], labels: dict[TrackKey, int]) -> float:
    keys = sorted(scores)
    y_true = [labels[k] for k in keys]
    candidates = sorted({scores[k] for k in keys})
    return max(candidates, key=lambda t: (f1_score(y_true, [scores[k] >= t for k in keys], zero_division=0), -t))


def out_of_fold_scores(features: np.ndarray, tracks: list[TrackKey], labels: dict[TrackKey, int],
                       folds: int) -> dict[TrackKey, float]:
    """Each track is scored by a probe that never saw any crop of that track."""
    groups = group_ids(tracks)
    y = np.array([labels[t] for t in tracks])
    scores: dict[TrackKey, float] = {}
    for train, test in GroupKFold(n_splits=folds).split(features, y, groups):
        probe = fit_probe(features[train], y[train])
        scores.update(mean_by_track([tracks[i] for i in test], probe.predict_proba(features[test])[:, 1]))
    return scores


def probe_cross_validated(features: np.ndarray, crops: Crops, labels: dict[TrackKey, int]) -> MethodResult:
    """Outer folds score the tracks; inside each, the threshold comes from the training tracks only."""
    rows = [i for i, track in enumerate(crops.tracks) if track in labels]
    x, tracks = features[rows], [crops.tracks[i] for i in rows]
    groups = group_ids(tracks)
    y = np.array([labels[t] for t in tracks])

    scores: dict[TrackKey, float] = {}
    is_staff: dict[TrackKey, bool] = {}
    for train, test in GroupKFold(n_splits=OUTER_FOLDS).split(x, y, groups):
        train_tracks = [tracks[i] for i in train]
        inner = out_of_fold_scores(x[train], train_tracks, labels, INNER_FOLDS)
        threshold = best_f1_threshold(inner, labels)
        probe = fit_probe(x[train], y[train])
        fold_scores = mean_by_track([tracks[i] for i in test], probe.predict_proba(x[test])[:, 1])
        scores.update(fold_scores)
        is_staff.update({track: score >= threshold for track, score in fold_scores.items()})
    return MethodResult("probe", scores, is_staff)


def rule_scores(config: StoreConfig) -> MethodResult:
    """Share of each billing track's foot points inside the staff area behind the counter."""
    camera = config.camera("billing")
    tracks = drop_short_tracks(group_by_track(read_tracks_csv(config.outputs_dir / "tracks" / "billing.csv")),
                               config.events.min_track_seconds)
    scores = {("billing", track_id): sum(point_in_polygon(s.foot, camera.staff_area) for s in samples) / len(samples)
              for track_id, samples in tracks.items()}
    return MethodResult("rule", scores, {key: score >= 0.5 for key, score in scores.items()})


@dataclass
class Score:
    method: str
    scope: str
    tracks: int
    staff: int
    precision: float
    recall: float
    customers_kept: float | None
    f1: float
    pr_auc: float | None


def score(result: MethodResult, labels: dict[TrackKey, int], scope: str, cameras: tuple[str, ...]) -> Score | None:
    keys = sorted(k for k in labels if k[0] in cameras and k in result.scores)
    if not keys:
        return None
    y_true = [labels[k] for k in keys]
    y_pred = [result.is_staff[k] for k in keys]
    both_classes = 0 < sum(y_true) < len(y_true)
    negatives = [p for t, p in zip(y_true, y_pred) if t == 0]
    return Score(
        method=result.name,
        scope=scope,
        tracks=len(keys),
        staff=sum(y_true),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        customers_kept=sum(not p for p in negatives) / len(negatives) if negatives else None,
        f1=f1_score(y_true, y_pred, zero_division=0),
        pr_auc=average_precision_score(y_true, [result.scores[k] for k in keys]) if both_classes else None,
    )


def comparison_table(scores: list[Score]) -> str:
    lines = ["| Method | Tracks (staff) | Precision | Recall | Non-staff kept | F1 | PR-AUC |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in scores:
        pr_auc = f"{s.pr_auc:.2f}" if s.pr_auc is not None else "n/a (one class only)"
        kept = f"{s.customers_kept:.2f}" if s.customers_kept is not None else "n/a"
        lines.append(f"| {s.method} | {s.tracks} ({s.staff}) | {s.precision:.2f} | {s.recall:.2f} | {kept} "
                     f"| {s.f1:.2f} | {pr_auc} |")
    return "\n".join(lines)


def final_probe(features: np.ndarray, crops: Crops, labels: dict[TrackKey, int],
                cross_validated: MethodResult) -> MethodResult:
    """The probe retrained on every labelled crop and applied to all tracks, labelled or not."""
    rows = [i for i, track in enumerate(crops.tracks) if track in labels]
    probe = fit_probe(features[rows], np.array([labels[crops.tracks[i]] for i in rows]))
    threshold = best_f1_threshold(cross_validated.scores, labels)
    scores = mean_by_track(crops.tracks, probe.predict_proba(features)[:, 1])
    return MethodResult("probe", scores, {key: s >= threshold for key, s in scores.items()})


def write_track_labels(result: MethodResult, path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["camera", "track_id", "staff_score", "is_staff", "method"])
        for (camera, track_id), value in sorted(result.scores.items()):
            writer.writerow([camera, track_id, round(value, 4), int(result.is_staff[(camera, track_id)]), result.name])


def write_scores(results: list[MethodResult], labels: dict[TrackKey, int], path: Path) -> None:
    keys = sorted(set().union(*(r.scores for r in results)))
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["camera", "track_id", "label", *[f"{r.name}_score" for r in results]])
        for key in keys:
            label = {1: "staff", 0: "not staff"}.get(labels.get(key), "unsure")
            writer.writerow([*key, label, *[round(r.scores[key], 4) if key in r.scores else "" for r in results]])


def main() -> None:
    argparse.ArgumentParser(description="Compare staff classifiers and label every track.").parse_args()
    config = load_config()
    crops = read_crops()
    labels = read_labels()

    model, preprocess, tokenizer = load_clip()
    features = embed_images(crops.paths, model, preprocess)
    zero_shot_crops = zero_shot_crop_scores(features, model, tokenizer)
    zero_shot_tracks = mean_by_track(crops.tracks, zero_shot_crops)
    zero_shot = MethodResult("zero-shot", zero_shot_tracks, {k: s >= 0.5 for k, s in zero_shot_tracks.items()})
    probe = probe_cross_validated(features, crops, labels)
    rule = rule_scores(config)

    scopes = [("all cameras", ("entry", *IN_STORE_CAMERAS)), ("in-store cameras", IN_STORE_CAMERAS),
              ("billing camera", ("billing",))]
    sections = []
    for scope, cameras in scopes:
        rows = [s for s in (score(m, labels, scope, cameras) for m in (rule, zero_shot, probe)) if s]
        sections += [f"### {scope.capitalize()}", "", comparison_table(rows), ""]

    in_store = {r.name: score(r, labels, "in-store", IN_STORE_CAMERAS) for r in (zero_shot, probe)}
    winner = max(in_store, key=lambda name: in_store[name].f1)
    final = final_probe(features, crops, labels, probe) if winner == "probe" else zero_shot

    write_scores([rule, zero_shot, probe], labels, config.outputs_dir / "staff_scores.csv")
    write_track_labels(final, config.outputs_dir / "track_labels.csv")

    staff_tracks = sum(final.is_staff.values())
    in_store_staff = [probe.scores[k] for k, v in labels.items() if v == 1 and k[0] in IN_STORE_CAMERAS]
    customers = [probe.scores[k] for k in read_customer_tracks() if k in probe.scores]
    report = "\n".join([
        "# Staff classification",
        "",
        f"Labelled tracks: {len(labels)} ({sum(labels.values())} staff), from data/labels/staff.csv (labelled from",
        "crops and frame context; unsure tracks left out). Scores are per track, the mean over its crops.",
        "The probe's numbers are out-of-fold: 5 folds grouped by track, with each fold's threshold chosen by inner",
        "cross-validation on its training tracks only. The rule and zero-shot learn nothing from the labels.",
        "",
        *sections,
        "## Decision",
        "",
        f"Winner on the in-store cameras by F1: **{winner}** (zero-shot {in_store['zero-shot'].f1:.2f}, "
        f"probe {in_store['probe'].f1:.2f}). The rule only covers the billing camera, where every labelled track "
        "is staff, so it cannot be measured on customers there.",
        "",
        f"The weak spot is in-store customers: the probe keeps only {in_store['probe'].customers_kept:.0%} of the "
        f"{in_store['probe'].tracks - in_store['probe'].staff} in-store non-staff tracks (zero-shot "
        f"{in_store['zero-shot'].customers_kept:.0%}). Customers here dress like the black staff uniform, and there are "
        "few in-store customer examples, so it leans towards 'staff'. Customer-only metrics are therefore a lower "
        "bound, and the report shows all-people numbers next to them.",
        "",
        f"No threshold fixes it: the {len(customers)} labelled in-store customer tracks score "
        f"{min(customers):.2f} to {max(customers):.2f}, inside the staff tracks' range of {min(in_store_staff):.2f} "
        f"to {max(in_store_staff):.2f}. They come from only two people, so the probe barely saw what a customer "
        "looks like here. More labelled customers, ideally from other days, is the fix.",
        "",
        f"Applied to all {len(final.scores)} tracks: {staff_tracks} staff, {len(final.scores) - staff_tracks} not staff "
        "(outputs/track_labels.csv).",
        "",
        "Caveat: a person can appear in several track IDs, and grouping is by track, not by person, so a person",
        "seen in a training track can reappear in a test track. The probe's numbers are somewhat optimistic.",
    ])
    (config.outputs_dir / "staff_comparison.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
