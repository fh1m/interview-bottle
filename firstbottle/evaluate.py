"""Re-examine the 2023 interview model with the data it was trained on.

  1. The training run, read back from its own TensorBoard event files.
  2. AP@0.5 on the 15-image test split it never trained on.
  3. The same test split, sunk through the water model from fh1m/green-at-depth:
     would a bottle detector trained on desk photos have survived a pool?

    python -m firstbottle.evaluate     # prints the numbers, writes assets/*.png
"""

from __future__ import annotations

import glob
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "workspace"
ASSETS = ROOT / "assets"
MODEL = WS / "exported_models/ssd_mobilenet/saved_model"

# Same physics as green-at-depth (Pope & Fry pure water x4, BGR order).
K_BGR = np.array([0.0094, 0.0447, 0.2224]) * 4.0
VEIL_BGR = np.array([0.55, 0.42, 0.08])


def submerge_rgb(rgb: np.ndarray, d: float) -> np.ndarray:
    bgr = rgb[..., ::-1].astype(float) / 255
    t = np.exp(-K_BGR * d)
    out = bgr * t + VEIL_BGR * (1 - t)
    return np.clip(out[..., ::-1] * 255, 0, 255).astype(np.uint8)


def events(pattern: str) -> dict[str, list[tuple[int, float]]]:
    tags: dict[str, list[tuple[int, float]]] = {}
    for f in sorted(glob.glob(str(WS / pattern), recursive=True)):
        for rec in tf.data.TFRecordDataset(f):
            e = event_pb2.Event.FromString(rec.numpy())
            for v in e.summary.value:
                try:
                    val = float(tf.make_ndarray(v.tensor)) if v.HasField("tensor") else float(v.simple_value)
                except (TypeError, ValueError):
                    continue
                tags.setdefault(v.tag, []).append((e.step, val))
    return tags


def load_split(name: str):
    feats = {"image/encoded": tf.io.FixedLenFeature([], tf.string)}
    for k in ("xmin", "ymin", "xmax", "ymax"):
        feats[f"image/object/bbox/{k}"] = tf.io.VarLenFeature(tf.float32)
    out = []
    for r in tf.data.TFRecordDataset(str(WS / f"data/{name}.tfrecord")):
        e = tf.io.parse_single_example(r, feats)
        img = tf.io.decode_image(e["image/encoded"], channels=3).numpy()
        box = np.stack([tf.sparse.to_dense(e[f"image/object/bbox/{k}"]).numpy()
                        for k in ("ymin", "xmin", "ymax", "xmax")], axis=1)
        out.append((img, box))
    return out


def iou(a, b):
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    y2, x2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def average_precision(model, data, depth=0.0) -> float:
    """VOC-style AP@0.5 (all-point interpolation), one class."""
    scored, n_gt = [], 0
    for img, gt in data:
        x = submerge_rgb(img, depth) if depth else img
        det = model(tf.convert_to_tensor(x[None], tf.uint8))
        boxes, scores = det["detection_boxes"][0].numpy(), det["detection_scores"][0].numpy()
        used = np.zeros(len(gt), bool)
        n_gt += len(gt)
        for b, s in sorted(zip(boxes, scores), key=lambda p: -p[1]):
            if s < 0.01:
                break
            ious = [iou(b, g) for g in gt]
            j = int(np.argmax(ious)) if ious else -1
            hit = j >= 0 and ious[j] >= 0.5 and not used[j]
            if hit:
                used[j] = True
            scored.append((s, hit))
    scored.sort(key=lambda p: -p[0])
    tp = np.cumsum([h for _, h in scored])
    fp = np.cumsum([not h for _, h in scored])
    rec, prec = tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 10})

    # 1. the training run, as it actually happened
    tr = events("models/ssd_mobilenet/train/events*")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), dpi=150, sharex=True)
    for tag, c in (("Loss/localization_loss", "#1f4e8c"), ("Loss/classification_loss", "#c2571a"),
                   ("Loss/regularization_loss", "#b3261e"), ("Loss/total_loss", "#333")):
        s, v = zip(*tr[tag])
        a1.plot(s, v, color=c, lw=1.8, label=tag.split("/")[1].replace("_loss", ""))
    s, v = zip(*tr["learning_rate"])
    a2.plot(s, v, color="#2a6f4e", lw=2)
    a1.set_ylabel("loss"); a1.legend(frameon=False, fontsize=8, ncol=2)
    a2.set_ylabel("learning rate"); a2.set_xlabel("training step")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(ASSETS / "training_2023.png"); plt.close(fig)

    ev = events("models/ssd_mobilenet/eval/**/events*")
    logged = ev["DetectionBoxes_Precision/mAP@.50IOU"][-1]

    # 2 + 3. AP@0.5, dry and wet.
    # test.tfrecord turned out to be a byte-for-byte copy of train.tfrecord, so
    # the honest held-out set is validation (10 images, never trained on).
    model = tf.saved_model.load(str(MODEL))
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    leaked = sum(any(np.array_equal(a, b) for b, _ in train) for a, _ in test)
    ap_test = average_precision(model, test)
    depths = [0, 1, 2, 3, 5, 8]
    aps = [average_precision(model, val, d) for d in depths]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(depths, aps, "o-", color="#1f4e8c", lw=2, label="validation (held out)")
    ax.axhline(ap_test, color="#b3261e", lw=1, ls="--", label="'test' split (= training images)")
    ax.set_xlabel("water between camera and bottle (m), green-at-depth model")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(ASSETS / "ap_vs_depth.png"); plt.close(fig)

    # contact sheet: one test image, dry and at 3 m, with the top box
    sheet = []
    for img, gt in val[:4]:
        row = []
        for d in (0, 3):
            x = submerge_rgb(img, d) if d else img.copy()
            det = model(tf.convert_to_tensor(x[None], tf.uint8))
            b, s = det["detection_boxes"][0].numpy()[0], float(det["detection_scores"][0].numpy()[0])
            h, w = x.shape[:2]
            vis = np.ascontiguousarray(x[..., ::-1])
            if s > 0.3:
                cv2.rectangle(vis, (int(b[1] * w), int(b[0] * h)), (int(b[3] * w), int(b[2] * h)), (0, 255, 0), 3)
            cv2.putText(vis, f"{d} m  score {s:.2f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            row.append(cv2.resize(vis, (320, 320)))
        sheet.append(np.hstack(row))
    cv2.imwrite(str(ASSETS / "dry_vs_3m.png"), np.vstack([np.hstack(sheet[:2]), np.hstack(sheet[2:4])]))

    # The habit: how much of the frame does the top box cover, vs the true box?
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    pred_cov, true_cov = [], []
    for img, gt in val:
        det = model(tf.convert_to_tensor(img[None], tf.uint8))
        pred_cov.append(area(det["detection_boxes"][0].numpy()[0]))
        true_cov.append(max(area(g) for g in gt))
    gt_train = [max(area(g) for g in gt) for _, gt in train]
    print(f"training: {tr['Loss/total_loss'][-1][0]} steps logged of 50000 configured")
    print(f"learning rate at last log: {tr['learning_rate'][-1][1]:.3f}")
    rl = tr["Loss/regularization_loss"]
    print(f"regularization loss: {rl[0][1]:.3f} -> {rl[-1][1]:.3f}")
    print(f"logged eval mAP@0.5 at step {logged[0]}: {logged[1]:.3f} (validation, 10 images)")
    print(f"'test' split: {leaked} of {len(test)} images are byte-identical to training images; AP@0.5 = {ap_test:.3f}")
    for d, a in zip(depths, aps):
        print(f"AP@0.5 validation at {d} m: {a:.3f}")
    print(f"top box covers {np.median(pred_cov) * 100:.0f}% of the frame (median); "
          f"the real bottle covers {np.median(true_cov) * 100:.0f}% (validation), "
          f"{np.median(gt_train) * 100:.0f}% in training")


if __name__ == "__main__":
    main()
