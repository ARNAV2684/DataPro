# Image Pipeline

The image data flow runs through the same five stages as the numeric/text flows:
**upload → preprocess → augment → EDA → model**. This document describes the
data format, the backend endpoints, and how the frontend connects to them.

## Data format

An image "dataset" is one of:

- **A ZIP archive of images**, optionally organised into **class sub-folders**
  (ImageFolder layout). The immediate parent folder of each image is treated as
  its class label, e.g.:

  ```
  dataset.zip
  ├── cats/   cat1.jpg, cat2.jpg, ...
  └── dogs/   dog1.jpg, dog2.jpg, ...
  ```

- **A single image file** (no label).

Class labels (folder names) are **required for model training** (needs ≥ 2
classes). Preprocessing, augmentation and most EDA work with or without labels.

Every stage downloads the input, processes **all** images, and writes the result
back as a ZIP to the next storage bucket — keeping the existing
"one storage key per artifact" model. Supported image types: `.png .jpg .jpeg
.bmp .gif .webp .tiff`.

## Shared code

- **`imagelib/`** — `load_images()` / `save_images_zip()` (ZIP or single image,
  preserving class folders), `augment_each()`, lenient parameter lookup
  (`get_param`), and CLI/EDA scaffolding.
- **`api/shared/image_pipeline.py`** — route helpers: `parse_summary`,
  `extract_dataset_id`, `download_image_input` (tries `preprocessed` →
  `augmented` → `datasets`).

Processing scripts are standalone CLI programs (run by `module_runner` as
subprocesses) and live alongside the numeric/text equivalents:
`2and3week/PPimage/`, `4and5week/Augmentation/imageaug/`, `EDA/image_EDA/`,
`6week/models/image_classifier.py`.

## Endpoints

Each stage exposes **one consolidated image endpoint**. The specific operation
is chosen by a field in the request body (so the frontend dispatches by id).

### 1. Preprocess — `POST /api/preprocess/image`

Body: `PreprocessRequest` (`user_id`, `dataset_key`, `operation`, `params`).
Output ZIP → `preprocessed` bucket.

| `operation`        | Effect | Key params |
|--------------------|--------|------------|
| `image-validation` | Report formats/sizes/classes, drop corrupt images | – |
| `resize-normalize` | Resize to target size (default 224×224) | `targetSize`/`width`+`height`, `grayscale`, `normalization` |
| `color-correction` | Brightness/contrast/saturation (multipliers) | `brightness`, `contrast`, `saturation` |
| `noise-reduction`  | Gaussian or median denoise | `method` (`gaussian`/`median`), `strength` (0–1) |
| `format-conversion`| Convert colour mode | `format` (`RGB`/`grayscale`/`RGBA`), `quality` |

### 2. Augment — `POST /api/augment/image`

Body: `AugmentRequest` (`user_id`, `dataset_key`, `technique`, `params`).
Grows the dataset (originals + `_aug`/`_mix` copies) → `augmented` bucket.

| `technique`         | Effect | Key params |
|---------------------|--------|------------|
| `rotation`          | Random rotation | `Max Angle`, `copies` |
| `color-jitter`      | Random brightness/contrast/saturation | `Brightness`, `Contrast`, `Saturation` |
| `elastic-transform` | Smooth elastic deformation (scipy) | `Alpha`, `Sigma` |
| `cutout`            | Mask random squares | `Hole Size`, `Number of Holes` |
| `mixup`             | Blend same-class image pairs | `Alpha` |

### 3. EDA — `POST /api/eda/image`

Body: `EDARequest` (`user_id`, `dataset_key`, `analysis_type`, `params`).
`analysis_type` is matched case-insensitively (spaces → hyphens). Generates
visualization PNGs → `eda` bucket; their public URLs are returned in
`meta.visualization_urls` (`[{key, url, filename}]`).

| `analysis_type`        | Output |
|------------------------|--------|
| `image-statistics`     | Images-per-class bar chart, resolution scatter, stats |
| `color-analysis`       | RGB histograms, mean image, average colour |
| `quality-assessment`   | Sharpness + brightness distributions, blur/exposure flags |
| `feature-extraction`   | PCA 2D embedding of thumbnails, coloured by class |
| `similarity-analysis`  | Cosine similarity heatmap, most-similar pair |
| `object-detection`     | Edge-based region bounding boxes on sample images |

### 4. Model — `POST /api/model/image`

Body: `ModelRequest` (`user_id`, `dataset_key`, `model_type`, `hyperparameters`,
`validation_split`). Trains a classifier, uploads `model.pt`, `label_map.json`,
`metrics.json`, `training_curve.png` → `models` bucket. Returns `metrics`
(accuracy/precision/recall/F1) and `model_id`.

| `model_type`         | Architecture |
|----------------------|--------------|
| `cnn-basic`          | Small CNN from scratch (`Conv Layers`, `Filters`, `Dropout`) |
| `resnet`             | torchvision ResNet18/34/50 (`Architecture`, `Pretrained`) |
| `efficientnet`       | EfficientNet-B0 (`Pretrained`) |
| `vision-transformer` | ViT-B/16 (`Pretrained`) |

Pretrained weights download on first use (~45 MB for ResNet18); the trainer
falls back to random initialisation automatically if offline. Training is
CPU-friendly: default 5 epochs, images capped per class (`--max-per-class`).

## Frontend wiring

The image UI already existed; these methods connect it:

- `apiClient`: `preprocessImage`, `augmentImage`, `runImageEDA`,
  `trainImageModel`, and `image` added to the upload `data_type`.
- `HomePage`: single images and `.zip` archives are now uploaded with
  `data_type: 'image'`.
- `PreprocessingPage` / `DataAugmentationPage` / `ExploratoryDataAnalysisPage` /
  `ModelTrainingPage`: each dispatches its image options to the endpoints above.

## Dependencies

Added to `api/requirements.txt`: `torchvision`, `Pillow`, `matplotlib`
(`torch`, `numpy`, `scikit-learn`/`scipy` were already present).
