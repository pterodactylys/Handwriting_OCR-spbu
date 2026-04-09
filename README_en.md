## [English](README_en.md) | [Русский](README.md)

## Handwritten Text Recognition with Sentiment Analysis

## Project Description

The goal is to develop software that:
1. Accepts an image with handwritten text in Russian or English.
2. Converts it into digital text using a deep learning model (CRNN + CTC).
3. Analyzes the emotional tone (sentiment) of the recognized text using a lexicon-based approach.

The main focus of the project is implementing handwritten text recognition algorithms. Sentiment analysis is a secondary task.

## Features (preliminary plan, may be adjusted during development)

- **Image preprocessing** - binarization, line alignment, resizing, normalization.
- **Handwritten text recognition** - convolutional recurrent neural network (CRNN) with CTC loss.
- **Sentiment analysis** - lexicon-based method using ready-made dictionaries.
- **Web interface** for uploading images and viewing results.

## Dataset

- **English** For training and evaluation, the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) will be used. It contains images of handwritten English texts with page-level and line-level annotations (transcriptions).

- **Russian** For training, the [HTR Dataset](https://link.springer.com/article/10.1007/s11042-021-11399-6) will be used.

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/pterodactylys/Handwriting_OCR-spbu.git
	cd Handwriting_OCR-spbu
	```

2. Create a virtual environment (recommended):

	```bash
	python -m venv venv
	source venv/bin/activate  # Linux/Mac
	venv\Scripts\activate     # Windows
	```

3. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

## Project Structure (approximate)

```text
Handwriting_OCR-spbu/
├── data/               # Directory with datasets (not included in this repository; access can be obtained via the links above)
│   ├── hkr/            # HKR Dataset
│   ├── lexicons/       # Image transformations
│   └── iam/            # IAM Database
├── models/             # Neural network models
│   ├── crnn.py         # CRNN architecture
│   └── ctc_decoder.py  # CTC decoding
├── train.ipynb         # Notebook with the full workflow
├── train_ru.ipynb      # Same, but for Russian
├── requirements.txt    # Python dependencies
└── README.md
```

## Results

Will be added after the first working version of the program is implemented.

## Authors

- Andrey Telitsin [@atelitsin](https://github.com/atelitsin)
- Kamil Shaimiev [@kameel4](https://github.com/kameel4)
