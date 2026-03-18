# YOLO Object Detection Training Pipeline

[Portugues](#portugues) | [English](#english)

---

## English

### Overview

End-to-end YOLO object detection pipeline with custom dataset creation, model training, evaluation, and inference. Built with Ultralytics YOLOv8 for state-of-the-art real-time object detection.

**DIO Lab Project** - Formacao Machine Learning Specialist

### Features

- **Dataset Creation**: Tools for annotation, format conversion (VOC, COCO, YOLO)
- **Data Augmentation**: Random crop, flip, mosaic, mixup, color jitter
- **Model Training**: Fine-tune YOLOv8 (nano to xlarge) on custom datasets
- **Evaluation**: mAP@50, mAP@50-95, precision, recall per class
- **Inference**: Image, video, and webcam real-time detection
- **Export**: ONNX, TensorRT, CoreML model export

### Tech Stack

- Python 3.10+
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- Docker
- GitHub Actions CI/CD

### Project Structure

```
python-yolo-object-detection/
|-- src/
|   |-- __init__.py
|   |-- dataset_manager.py
|   |-- train.py
|   |-- evaluate.py
|   |-- inference.py
|-- tests/
|   |-- __init__.py
|   |-- test_dataset.py
|   |-- test_inference.py
|-- data/
|   |-- sample/
|-- configs/
|   |-- training_config.yaml
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|-- Dockerfile
|-- requirements.txt
|-- README.md
|-- LICENSE
```

### Quick Start

```bash
git clone https://github.com/galafis/python-yolo-object-detection.git
cd python-yolo-object-detection
pip install -r requirements.txt
python -m src.train --config configs/training_config.yaml
```

### Docker

```bash
docker build -t yolo-detection .
docker run --rm yolo-detection
```

### License

MIT License - see [LICENSE](LICENSE).

---

## Portugues

### Visao Geral

Pipeline completo de deteccao de objetos YOLO com criacao de dataset customizado, treinamento, avaliacao e inferencia. Construido com Ultralytics YOLOv8 para deteccao em tempo real.

**Projeto Lab DIO** - Formacao Machine Learning Specialist

### Funcionalidades

- **Criacao de Dataset**: Ferramentas de anotacao e conversao de formatos
- **Aumento de Dados**: Crop, flip, mosaico, mixup, variacao de cor
- **Treinamento**: Fine-tune YOLOv8 em datasets customizados
- **Avaliacao**: mAP@50, mAP@50-95, precisao, recall por classe
- **Inferencia**: Deteccao em imagens, video e webcam em tempo real
- **Exportacao**: ONNX, TensorRT, CoreML

### Inicio Rapido

```bash
git clone https://github.com/galafis/python-yolo-object-detection.git
cd python-yolo-object-detection
pip install -r requirements.txt
python -m src.train --config configs/training_config.yaml
```

### Licenca

Licenca MIT - veja [LICENSE](LICENSE).
