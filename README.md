# IAM Offline HTR Baseline

This repository contains a teaching-oriented baseline for offline handwritten text recognition (HTR) on the IAM Handwriting Database.

The current scope is intentionally narrow:
- input: one line image
- output: one text line
- model: CNN + BiLSTM + CTC
- decoder: greedy CTC decoding

The project does not include page segmentation, beam search, language models, attention, transformers, or emotion analysis.

## Project layout

```text
Handwriting_OCR-spbu/
|-- configs/
|   `-- base.yaml
|-- data/
|   |-- ascii/                 # Present locally, but empty in this workspace
|   |-- formsA-D/              # Full form images
|   |-- lines/                 # Line images used for HTR
|   |-- words/                 # Word images, not used by the baseline
|   |-- xml/                   # XML annotations with line ids, texts, writer ids
|   |-- iam_manifest_train.csv
|   |-- iam_manifest_val.csv
|   `-- iam_manifest_test.csv
|-- scripts/
|   `-- prepare_iam.py
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- decode.py
|   |-- eval.py
|   |-- infer.py
|   |-- metrics.py
|   |-- model.py
|   |-- tokenizer.py
|   |-- train.py
|   `-- utils.py
|-- artifacts/
|   `-- alphabet.json          # Created during training
|-- checkpoints/
|   `-- crnn_ctc_baseline/     # Created during training
|-- models/
|-- utils/
`-- requirements.txt
```

## Dataset structure found locally

The baseline was adapted to the dataset layout that already exists in this repository.

Relevant paths:
- `data/formsA-D/*.png`: full-page form images
- `data/lines/<prefix>/<form_id>/<line_id>.png`: line images for recognition
- `data/words/<prefix>/<form_id>/<word_id>.png`: word crops, not used here
- `data/xml/<form_id>.xml`: XML annotations with handwritten line metadata

Important observation:
- `data/ascii/` exists but is empty in this workspace
- no official IAM split files were found locally

Because official split files are not present, the baseline creates a deterministic split by `writer-id` from XML. This avoids putting the same handwriting style in both train and validation/test.
After the initial split, the preparation script moves the minimum number of writers into train if needed so that every character seen in validation/test is also present in the train alphabet.

## How line text is matched to images

For each line image:
1. The image file name gives the line id, for example `a01-003-04.png`.
2. The corresponding XML file is `data/xml/a01-003.xml`.
3. The script finds `<line id="a01-003-04" text="...">` inside that XML.
4. The target text is taken from the XML `text` attribute.
5. The writer is taken from `<form writer-id="...">` and used for the split.

The preparation script also normalizes text by:
- decoding escaped entities such as `&quot;`
- trimming whitespace
- collapsing repeated whitespace into single spaces

By default, XML lines with `segmentation="err"` are dropped because IAM explicitly marks them as problematic.

## Model

The recognition model is a compact CRNN:

1. grayscale line image
2. resize to fixed height, preserve aspect ratio
3. CNN feature extractor
4. convert feature map into a left-to-right sequence over width
5. 2-layer bidirectional LSTM
6. linear classifier over characters
7. `log_softmax`
8. CTC loss during training

Decoding uses plain greedy CTC decoding:
- `argmax`
- remove repeated symbols
- remove blank token

## Metrics

The project reports:
- CER: character error rate
- WER: word error rate

Both are based on edit distance:
- CER compares character sequences
- WER compares whitespace-separated word sequences

Lower is better.

## 1. Prepare manifests

Run:

```bash
python scripts/prepare_iam.py
```

This produces:
- `data/iam_manifest_train.csv`
- `data/iam_manifest_val.csv`
- `data/iam_manifest_test.csv`

Each CSV has the format:

```text
image_path,text
data/lines/a01/a01-003/a01-003-00.png,Though they may gather some Left-wing support, a
```

Useful options:

```bash
python scripts/prepare_iam.py --include-segmentation-errors
python scripts/prepare_iam.py --seed 123
python scripts/prepare_iam.py --skip-image-verification
```

The script prints:
- how many samples were kept
- how many were dropped and why
- split sizes
- average/min/max text lengths
- unique character set per split

## 2. Train

Main config:
- `configs/base.yaml`

Start training:

```bash
python -m src.train --config configs/base.yaml
```

What training does:
- builds the alphabet from the train manifest
- saves `artifacts/alphabet.json`
- saves `checkpoints/crnn_ctc_baseline/config.yaml`
- saves `checkpoints/crnn_ctc_baseline/alphabet.json`
- trains with Adam + CTC
- clips gradients
- tracks validation CER and WER
- saves:
  - `best.pt` by validation CER
  - `last.pt`
- stops early if validation CER does not improve

Sanity checks included in training:
- one batch forward pass before training
- one CTC loss computation before training
- greedy decode check before training
- optional overfit mode on a small subset
- validation prediction samples saved every epoch

To test if the pipeline can overfit a tiny subset, change in `configs/base.yaml`:

```yaml
overfit_num_samples: 32
train_augmentations: false
```

## 3. Evaluate

Run:

```bash
python -m src.eval --checkpoint checkpoints/crnn_ctc_baseline/best.pt
```

Optional overrides:

```bash
python -m src.eval \
  --checkpoint checkpoints/crnn_ctc_baseline/best.pt \
  --test-csv data/iam_manifest_test.csv \
  --output-csv artifacts/test_predictions.csv
```

The evaluation script:
- loads the checkpoint
- loads the saved alphabet
- evaluates on the test split
- prints final loss, CER, and WER
- saves predictions to CSV

Prediction CSV format:
- `image_path`
- `ground_truth`
- `prediction`

## 4. Inference on one image

Run:

```bash
python -m src.infer --checkpoint checkpoints/crnn_ctc_baseline/best.pt --image path/to/line.png
```

With debug output:

```bash
python -m src.infer \
  --checkpoint checkpoints/crnn_ctc_baseline/best.pt \
  --image data/lines/a01/a01-003/a01-003-00.png \
  --debug
```

Debug mode prints:
- source image path
- original size
- resized size
- sequence length after the CNN

## Config fields

Key fields in `configs/base.yaml`:
- `train_csv`, `val_csv`, `test_csv`: manifest files
- `img_height`: target input height
- `batch_size`
- `num_workers`
- `epochs`
- `lr`
- `weight_decay`
- `use_amp`
- `train_augmentations`
- `checkpoint_dir`
- `alphabet_path`
- `seed`
- `clip_grad_norm`
- `scheduler_patience`
- `early_stopping_patience`
- `overfit_num_samples`
- `model.hidden_size`
- `model.lstm_layers`
- `model.dropout`

## Notes about the current workspace

This repository already contains the IAM dataset locally, so no dataset download is needed.

During implementation, the local Python environment did not have `torch` installed. Because of that:
- manifest generation can be run directly
- training and inference code is implemented, but actual model training must be run in an environment where PyTorch is available

## Main commands

Prepare manifests:

```bash
python scripts/prepare_iam.py
```

Train:

```bash
python -m src.train --config configs/base.yaml
```

Evaluate:

```bash
python -m src.eval --checkpoint checkpoints/crnn_ctc_baseline/best.pt
```

Infer:

```bash
python -m src.infer --checkpoint checkpoints/crnn_ctc_baseline/best.pt --image path/to/line.png
```
