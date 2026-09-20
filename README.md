<div align="center">

# 🧠 Real-Time Facial Emotion Recognition

### Deep Learning • Cross-Dataset Generalisation • Explainable AI • Real-Time Inference

**Master's Dissertation Project**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Real--Time%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Pro-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Research](https://img.shields.io/badge/Project-Master's%20Dissertation-6f42c1)](#)

**A comparative investigation of CNN, ResNet50 and EfficientNet-B0 for facial emotion recognition, with particular emphasis on cross-dataset robustness and real-time deployment.**

</div>

---

## 🔍 Project at a Glance

This repository contains the complete experimental pipeline developed for my Master's dissertation on **real-time facial emotion recognition (FER)**.

Rather than evaluating a model only on the dataset used during development, this project investigates FER from three connected perspectives:

> 🎯 **Predictive Performance**  
> How accurately can deep-learning models recognise facial-expression categories?

> 🌍 **Cross-Dataset Generalisation**  
> Does a model trained on AffectNet remain reliable when evaluated on FERPlus?

> ⚡ **Real-Time Feasibility**  
> Can the selected model perform inference quickly enough for real-time applications?

The project compares a **custom CNN**, **ResNet50**, and **EfficientNet-B0**, investigates class-weighted learning, evaluates the selected model without target-domain adaptation on FERPlus, analyses failure patterns using **Grad-CAM**, and benchmarks inference on CPU and GPU.

---

## ⭐ Headline Results

<div align="center">

| 🎯 Metric | 📊 Result |
|:---|:---:|
| Best Validation Accuracy | **59.50%** |
| AffectNet Test Accuracy | **59.42%** |
| AffectNet Macro-F1 | **55.17%** |
| FERPlus Accuracy | **9.23%** |
| FERPlus Macro-F1 | **7.25%** |
| Cross-Dataset Accuracy Drop | **50.19 pp** |
| T4 GPU Mean Latency | **7.41 ms** |
| Theoretical Model Throughput | **134.97 inferences/s** |

</div>

### 💡 Central Finding

> **The selected model was computationally fast enough for real-time GPU inference, but its strong cross-dataset performance degradation shows that inference speed and in-domain accuracy alone are not sufficient indicators of practical robustness.**

This distinction between **in-domain performance**, **cross-domain generalisation**, and **computational efficiency** is the central theme of the dissertation.

---

## 🧪 Experimental Pipeline

```text
              ┌──────────────────────────────┐
              │   AffectNet Processed Subset │
              │        30,626 Images         │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   EDA & Data Preprocessing   │
              │  RGB • 224×224 • Augmentation│
              └──────────────┬───────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │          Architecture Comparison     │
          │                                      │
          │  Custom CNN • ResNet50 • EfficientNet│
          └──────────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Fine-Tuned ResNet50        │
              │   Best Validation: 59.50%    │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Class-Weighted Refinement    │
              │ Accuracy: 59.42%             │
              │ Macro-F1: 55.17%             │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┴───────────────┐
              │                              │
              ▼                              ▼
    ┌───────────────────┐         ┌────────────────────┐
    │ FERPlus External  │         │ Grad-CAM & Error   │
    │ Evaluation        │         │ Analysis           │
    │ Accuracy: 9.23%   │         └──────────┬─────────┘
    └─────────┬─────────┘                    │
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  Real-Time Benchmarking      │
              │  T4 GPU: 7.41 ms mean       │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ OpenCV + TorchScript Webcam  │
              │      FER Application         │
              └──────────────────────────────┘
```

---

## 🎭 Emotion Classes

The same eight-class mapping is maintained throughout the experimental pipeline:

<div align="center">

| 😠 Anger | 😒 Contempt | 🤢 Disgust | 😨 Fear |
|:---:|:---:|:---:|:---:|
| 😊 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise |

</div>

> Emoji are used here only as visual labels. Model training and evaluation use the fixed textual class mapping.

---

## 📚 Datasets

### AffectNet — Primary Dataset

The primary experiments use a **processed Kaggle subset of AffectNet** rather than the complete original AffectNet corpus.

| Split Source | Images |
|:---|---:|
| Original Training Folder | 16,108 |
| Untouched Test Folder | 14,518 |
| **Total** | **30,626** |

The original training folder was divided using an **85/15 stratified training-validation split**, while the provided test folder remained untouched during model selection.

- Source resolution: **96 × 96**
- Number of classes: **8**
- Training imbalance ratio: approximately **2.52**
- Corrupted images detected: **0**

### FERPlus — External Evaluation

FERPlus was used to investigate **zero-adaptation cross-dataset performance**.

After majority-vote filtering:

```text
FERPlus PrivateTest
└── 2,729 evaluation images
```

No FERPlus examples were used to train or fine-tune the selected AffectNet model before external evaluation.

> **Methodological note:** FERPlus is treated as a cross-dataset/cross-annotation evaluation based on FER2013 images, rather than as a completely independent image-acquisition domain.

---

## 🛠️ Image Preprocessing

All models use a consistent input pipeline:

```text
Input Image
    ↓
RGB Conversion
    ↓
Resize → 224 × 224
    ↓
Training Augmentation
    ↓
Tensor Conversion
    ↓
ImageNet Normalisation
    ↓
Neural Network
```

### Training Augmentation

```text
✓ Random horizontal flip       p = 0.5
✓ Random rotation              ±10°
✓ Brightness jitter            0.15
✓ Contrast jitter              0.15
✓ ImageNet normalisation
```

Aggressive cropping was deliberately avoided because the original images were only **96 × 96 pixels**.

---

# 🧠 Model Comparison

Three architecture families were evaluated under the same experimental pipeline.

### 1️⃣ Custom CNN

A purpose-built baseline:

```text
Input
 ↓
Conv 32
 ↓
Conv 64
 ↓
Conv 128
 ↓
Conv 256
 ↓
Adaptive Average Pooling
 ↓
Dropout
 ↓
8-Class Classifier
```

### 2️⃣ ResNet50

ImageNet-pretrained ResNet50 was evaluated in two stages:

```text
Stage 1 → Frozen backbone + train classification head

Stage 2 → Unfreeze layer4 + classification head
          ↓
       Fine-tuning
```

### 3️⃣ EfficientNet-B0

EfficientNet-B0 followed a similar procedure:

```text
Frozen Feature Extractor
        ↓
Classifier Training
        ↓
Final Feature Block Unfrozen
        ↓
Fine-Tuning
```

---

## 🏆 Architecture Results

| Architecture | Validation Accuracy |
|:---|---:|
| Custom CNN | 45.22% |
| ResNet50 — Frozen | 49.23% |
| 🥇 **ResNet50 — Fine-tuned** | **59.50%** |
| EfficientNet-B0 — Frozen | 48.86% |
| EfficientNet-B0 — Fine-tuned | 50.31% |

Fine-tuned ResNet50 outperformed:

- Custom CNN by **14.28 percentage points**
- Fine-tuned EfficientNet-B0 by **9.19 percentage points**

It was therefore selected for further optimisation.

---

## ⚖️ Class-Weighted Learning

Initial ResNet50 evaluation showed substantial differences between emotion classes.

To address moderate class imbalance, the selected model was refined using **class-weighted cross-entropy**.

### Before vs After Weighting

| Metric | Unweighted ResNet50 | Weighted ResNet50 |
|:---|---:|---:|
| Accuracy | 58.00% | **59.42%** |
| Macro-F1 | 52.95% | **55.17%** |
| Weighted-F1 | 56.57% | **58.46%** |

### Improvement

```text
Accuracy      +1.42 percentage points
Macro-F1      +2.22 percentage points
Class F1      Improved for 7 / 8 classes
```

The weighted ResNet50 became the **final selected model**.

---

## 🎯 Per-Class Results

| Emotion | Precision | Recall | F1 |
|:---|---:|---:|---:|
| Anger | 52.26% | 37.08% | 43.38% |
| Contempt | 54.77% | 58.69% | 56.66% |
| Disgust | 38.73% | 42.55% | 40.55% |
| Fear | 54.77% | 39.36% | 45.80% |
| 🥇 Happy | 84.30% | 86.39% | **85.33%** |
| 🥈 Neutral | 68.75% | 87.25% | **76.90%** |
| Sad | 48.86% | 58.21% | 53.13% |
| Surprise | 42.63% | 36.93% | 39.58% |

Happy and neutral were recognised most reliably, while anger, disgust, fear and surprise remained more challenging.

---

# 🌍 Cross-Dataset Generalisation

The final AffectNet-trained model was evaluated directly on **FERPlus without target-domain adaptation**.

### AffectNet → FERPlus

<div align="center">

| | AffectNet | FERPlus | Change |
|:---|---:|---:|---:|
| **Accuracy** | 🟢 **59.42%** | 🔴 **9.23%** | **−50.19 pp** |
| **Macro-F1** | 🟢 **55.17%** | 🔴 **7.25%** | **−47.92 pp** |

</div>

This was the largest performance change observed in the study.

### Prediction Behaviour

The external prediction distribution shifted heavily toward:

```text
Fear    ██████████████████████████████  1,543
Sad     ████████████████                  824
Anger   █████                             270
Other   ██
Happy                                      0
Neutral                                    0
```

Despite happy and neutral being the largest true FERPlus classes, the model did not predict any external samples as either category.

### Major Confusions

| True Emotion | Predicted | Cases |
|:---|:---|---:|
| Happy | Fear | **529** |
| Neutral | Fear | **442** |
| Neutral | Sad | **361** |
| Surprise | Fear | **307** |
| Happy | Sad | **225** |
| Sad | Fear | 112 |
| Neutral | Anger | 106 |
| Anger | Fear | 105 |

> **Key observation:** reasonable source-domain performance did not translate into reliable zero-adaptation performance on FERPlus.

---

# 🔬 Investigating the Domain Gap

Rather than treating the low FERPlus result as sufficient evidence by itself, additional diagnostic experiments were performed.

## 🎨 Grayscale Control

Because AffectNet was processed as RGB while FER2013/FERPlus images are grayscale, the AffectNet test set was converted to grayscale and evaluated using the same trained model.

| Condition | Accuracy | Macro-F1 |
|:---|---:|---:|
| 🟢 AffectNet RGB | **59.42%** | **55.17%** |
| 🟡 AffectNet Grayscale | **44.42%** | **40.97%** |
| 🔴 FERPlus | **9.23%** | **7.25%** |

### What happened?

```text
AffectNet RGB
59.42%
   │
   │ −15.00 pp
   ▼
AffectNet Grayscale
44.42%
   │
   │ −35.19 pp
   ▼
FERPlus
9.23%
```

Removing colour clearly affected performance.

However, grayscale conversion alone did **not** reproduce the full FERPlus degradation.

This suggests that colour differences contribute to the domain gap, while broader differences in image characteristics, resolution, expression distribution and annotation are also relevant.

---

# 🔥 Explainability with Grad-CAM

Grad-CAM was applied to the final ResNet50 to examine which spatial regions were associated with its predictions.

### Correct Predictions

Correct fear predictions often showed activation around:

```text
👁 Eyes
👁 Periocular region
👄 Mouth
🙂 Central facial region
```

### Incorrect Predictions

Interestingly, high-confidence errors such as:

```text
Surprise → Fear
Happy    → Fear
Neutral  → Fear
Sad      → Fear
```

could also show activation around plausible facial regions.

### Interpretation

> **The AffectNet-trained model frequently attends to semantically relevant facial regions, particularly the eyes and mouth, but the learned relationship between these visual features and emotion classes does not transfer reliably to FERPlus.**

Grad-CAM is used here as a **post-hoc localisation method**, not as causal evidence of why the model produced a prediction.

---

# ⚡ Real-Time Inference

The final weighted ResNet50 contains:

```text
Parameters       23,524,424
Checkpoint       90.05 MB
```

## 🟢 NVIDIA T4 GPU

| Metric | Result |
|:---|---:|
| Mean Latency | **7.41 ms** |
| Median | **5.72 ms** |
| P90 | **10.44 ms** |
| P95 | **13.50 ms** |
| P99 | **23.87 ms** |
| Theoretical Throughput | **134.97 inferences/s** |

## 💻 Colab CPU

| Metric | Result |
|:---|---:|
| Mean Latency | **131.43 ms** |
| Median | **115.92 ms** |
| Theoretical Throughput | **7.61 inferences/s** |

### GPU vs CPU

```text
Mean Model Latency

T4 GPU     █          7.41 ms
CPU        ██████████████████ 131.43 ms
```

The T4 results show that the **classifier itself is computationally compatible with real-time processing** on the tested GPU.

> ⚠️ **134.97 inferences/s is model-only throughput — not webcam FPS.**

Face detection, frame acquisition, preprocessing, drawing and display introduce additional computational overhead.

---

# 📹 Real-Time Webcam Application

The final model was exported using **TorchScript** and integrated into an OpenCV webcam pipeline.

```text
┌─────────┐
│ Webcam  │
└────┬────┘
     ▼
┌────────────────┐
│ Face Detection │
└────┬───────────┘
     ▼
┌────────────────┐
│   Face Crop    │
└────┬───────────┘
     ▼
┌────────────────┐
│ Preprocessing  │
│    224×224     │
└────┬───────────┘
     ▼
┌────────────────┐
│    ResNet50    │
└────┬───────────┘
     ▼
┌────────────────┐
│    Softmax     │
└────┬───────────┘
     ▼
┌──────────────────────┐
│ Emotion + Confidence │
└──────────────────────┘
```

The application displays:

- detected face;
- predicted expression category;
- prediction confidence;
- model latency;
- end-to-end frame rate.

---

# 📂 Repository Structure

```text
real-time-facial-emotion-recognition/
│
├── 📘 README.md
├── 📄 requirements.txt
├── 📄 LICENSE
├── 📄 .gitignore
│
├── 📓 notebooks/
│   ├── 01_Data_EDA_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Tuning_Evaluation.ipynb
│   ├── 04_External_Generalization_Explainability.ipynb
│   └── 05_RealTime_Final_Results.ipynb
│
├── 🧠 src/
│   ├── dataset.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── gradcam.py
│   └── utils.py
│
├── 📹 realtime/
│   ├── realtime_emotion_recognition.py
│   └── README.md
│
├── 📊 results/
│   ├── affectnet/
│   ├── ferplus/
│   ├── diagnostics/
│   ├── gradcam/
│   └── benchmarking/
│
├── 🖼️ figures/
├── 💾 models/
├── 📚 docs/
└── 🗂️ data/
```

---

# 📓 Notebooks

| Notebook | Purpose |
|:---|:---|
| `01_Data_EDA_Preprocessing.ipynb` | Dataset validation, EDA, splitting and preprocessing |
| `02_Model_Training.ipynb` | CNN, ResNet50 and EfficientNet-B0 training |
| `03_Tuning_Evaluation.ipynb` | Test evaluation and class-weighted refinement |
| `04_External_Generalization_Explainability.ipynb` | FERPlus, diagnostics and Grad-CAM |
| `05_RealTime_Final_Results.ipynb` | Benchmarking, export and deployment |

Together, the notebooks form the complete experimental workflow used in the dissertation.

---

# 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/real-time-facial-emotion-recognition.git
cd real-time-facial-emotion-recognition
```

### 2. Create a virtual environment

```bash
python -m venv emotion_env
```

### 3. Activate it

**Windows**

```bash
emotion_env\Scripts\activate
```

**macOS/Linux**

```bash
source emotion_env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the webcam application

```bash
python realtime/realtime_emotion_recognition.py
```

Press `q` to close the application.

---

# 📦 Data Availability

The datasets are **not redistributed through this repository**.

Users should obtain AffectNet/FER2013/FERPlus from their respective sources and comply with the associated licences and terms of use.

The repository provides the processing and evaluation code required to reconstruct the experimental workflow where permitted.

---

# ⚠️ Limitations

This research has several important limitations:

- the primary dataset is a **processed AffectNet subset**, not the complete AffectNet corpus;
- FERPlus is based on FER2013 images and therefore represents cross-dataset/cross-annotation evaluation rather than a completely independent acquisition domain;
- FERPlus PrivateTest is strongly imbalanced;
- contempt and disgust contain very few external test samples;
- only three architecture families were investigated;
- no target-domain adaptation was performed before FERPlus evaluation;
- Grad-CAM provides post-hoc localisation rather than causal explanations;
- CPU/GPU benchmarks are hardware-specific;
- static-image FER does not capture temporal facial behaviour.

---

# 🔭 Future Work

Potential extensions include:

- 🌍 domain adaptation and domain generalisation;
- 🧪 stronger cross-domain augmentation;
- ⚖️ focal loss and alternative imbalance-aware objectives;
- 📚 evaluation on additional independently acquired datasets;
- 📱 lightweight architectures for edge devices;
- ✂️ pruning and quantisation;
- 🎥 temporal video-based FER;
- 📊 uncertainty calibration;
- 🚫 confidence-based abstention under distribution shift;
- ⚡ broader end-to-end deployment benchmarking.

---

# 🧭 Responsible Use

Facial-expression classification should be interpreted carefully.

This project predicts **dataset-provided facial-expression categories** from images. It does not establish that a facial image provides an infallible measurement of a person's internal emotional or psychological state.

Real-world use of facial-analysis systems should consider:

- privacy and consent;
- dataset and demographic bias;
- uncertainty;
- distribution shift;
- consequences of incorrect predictions;
- the context in which predictions are used.

---

# 🧰 Technology Stack

<div align="center">

| Area | Technology |
|:---|:---|
| Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Architectures | CNN • ResNet50 • EfficientNet-B0 |
| Explainability | Grad-CAM |
| Data Processing | NumPy • pandas |
| Evaluation | scikit-learn |
| Development | Google Colab Pro |
| GPU Benchmark | NVIDIA T4 |
| Deployment | TorchScript + OpenCV |

</div>

---

# 📚 Key References

- **Mollahosseini, Hasani & Mahoor (2019)** — AffectNet
- **Barsoum et al. (2016)** — FERPlus
- **He et al. (2016)** — ResNet
- **Tan & Le (2019)** — EfficientNet
- **Selvaraju et al. (2017)** — Grad-CAM
- **Li & Deng (2022)** — Deep Facial Expression Recognition Survey
- **Chen et al. (2022)** — Cross-Domain Facial Expression Recognition
- **Li et al. (2023)** — Cross-Domain FER and Self-Training

Full bibliographic details are provided in the associated dissertation.

---

# 🎓 Dissertation

**Real-Time Facial Emotion Recognition Using Deep Learning:  
A Comparative and Cross-Domain Evaluation**

Master's Dissertation  
MSc Data Science / Artificial Intelligence  
**[Your University]**

The dissertation presents the complete methodology, literature review, experimental analysis, discussion, limitations and interpretation of the results contained in this repository.

---

# ✍️ Author

**[Your Name]**

MSc Data Science / Artificial Intelligence  
[University Name]

---

# 📖 Citation

If you use this repository or associated work in academic research, please cite:

```text
[Your Surname], [Initial]. ([Year]).
Real-Time Facial Emotion Recognition Using Deep Learning:
A Comparative and Cross-Domain Evaluation.
Master's Dissertation, [University Name].
```

---

<div align="center">

### ⭐ Real-time speed is only one part of reliable FER.

**59.42% AffectNet Accuracy • 9.23% FERPlus Accuracy • 7.41 ms T4 Inference**

---

Made as part of a Master's dissertation in Data Science / Artificial Intelligence.

</div>
