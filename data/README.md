This directory contains placeholder folders for dataset files.

data/raw/      ← Symlink or copy your GOOSE dataset here
data/processed/ ← Pre-extracted rare-class cutouts (for Copy-Paste augmentation)

These directories are excluded from git via .gitignore because dataset files
are too large to commit. Store them externally (e.g., Kaggle datasets, Google Drive).

Expected structure under data/raw/:
  goose_2d_train/
    images/train/       ← Training images (*.png)
    labels/train/       ← Label ID maps (*_labelids.png)
  goose_2d_val/
    images/val/
    labels/val/
  gooseEx_2d_train/
    images/
    labels/             ← Color-coded label PNGs (*_color.png)
  goose_label_mapping.csv
