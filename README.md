# YOLOv5 Salient Feature Guided Knowledge Distillation on a Lightweight YOLOv5n-attention-light Model

## Introduction

This repository contains the implementation of a novel approach for enhancing the performance of YOLOv5 models through Salient Feature Guided Knowledge Distillation (SFGKD). We introduce a lightweight variant of the YOLOv5n model, named YOLOv5n-attention-light, which incorporates attention mechanisms to improve detection accuracy while maintaining efficiency.

## Features

- **Salient Feature Guided Knowledge Distillation (SFGKD):** A technique to transfer knowledge from a larger, more accurate teacher model to a smaller, lightweight student model.
- **YOLOv5n-attention-light:** A new lightweight YOLOv5n model variant with integrated attention mechanisms to enhance feature extraction.
- **Efficient and Accurate:** Achieves a balance between model size, inference speed, and detection accuracy.

## Model Architecture

The YOLOv5n-attention-light model builds upon the YOLOv5n architecture with additional attention layers to focus on important features within the input images. The SFGKD method further refines the model by distilling critical knowledge from a pre-trained, larger teacher model.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/AbrarZahin247/YOLOv5_Salient_Map_Guided_Knowledge_Distillation.git && cd YOLOv5_Salient_Map_Guided_Knowledge_Distillation && git checkout light-attention
pip install -r requirements.txt
```

## Results
The YOLOv5n-attention-light model, trained with SFGKD, demonstrates improved detection accuracy on various benchmark datasets while maintaining a compact size and fast inference speed.

## Contributing
We welcome contributions to this project. Please feel free to submit issues, feature requests, or pull requests.