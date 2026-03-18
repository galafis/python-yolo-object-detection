"""YOLO Model Training Pipeline.

Train YOLOv8 models on custom datasets with configurable parameters.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Train YOLO models on custom datasets."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path) if config_path else self._default_config()

    def _default_config(self) -> Dict:
        """Return default training configuration."""
        return {
            "model": "yolov8n.pt",
            "data": "coco128.yaml",
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.01,
            "project": "runs/train",
            "name": "yolo_custom",
            "patience": 10,
            "save": True,
            "device": "",
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load training config from YAML."""
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        logger.warning(f"Config not found: {config_path}, using defaults")
        return self._default_config()

    def train(self) -> Dict:
        """Train the YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.config["model"])
            results = model.train(
                data=self.config["data"],
                epochs=self.config["epochs"],
                imgsz=self.config["imgsz"],
                batch=self.config["batch"],
                lr0=self.config["lr0"],
                project=self.config["project"],
                name=self.config["name"],
                patience=self.config["patience"],
                save=self.config["save"],
                device=self.config.get("device", ""),
            )
            logger.info("Training completed successfully")
            return {"status": "success", "results": str(results)}
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            return {"status": "error", "message": "ultralytics not installed"}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}

    def evaluate(self, model_path: Optional[str] = None) -> Dict:
        """Evaluate trained model."""
        try:
            from ultralytics import YOLO
            path = model_path or f"{self.config['project']}/{self.config['name']}/weights/best.pt"
            model = YOLO(path)
            metrics = model.val()
            results = {
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
            logger.info(f"Evaluation: mAP50={results['mAP50']:.4f}")
            return results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"status": "error", "message": str(e)}


def main():
    """Run YOLO training pipeline."""
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    trainer = YOLOTrainer(config_path=args.config)

    print("=" * 60)
    print("YOLO Object Detection Training Pipeline")
    print("=" * 60)
    print(f"\nConfig: {trainer.config}")

    if args.evaluate:
        results = trainer.evaluate()
    else:
        results = trainer.train()

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
