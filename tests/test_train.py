"""Tests for YOLO training pipeline."""

import pytest
from src.train import YOLOTrainer


class TestYOLOTrainer:
    """Test YOLO trainer class."""

    def test_default_config(self):
        """Test default configuration."""
        trainer = YOLOTrainer()
        assert trainer.config is not None
        assert "model" in trainer.config
        assert "epochs" in trainer.config
        assert "imgsz" in trainer.config
        assert trainer.config["model"] == "yolov8n.pt"

    def test_config_values(self):
        """Test default config values."""
        trainer = YOLOTrainer()
        assert trainer.config["epochs"] == 50
        assert trainer.config["imgsz"] == 640
        assert trainer.config["batch"] == 16

    def test_nonexistent_config(self):
        """Test loading nonexistent config file."""
        trainer = YOLOTrainer(config_path="nonexistent.yaml")
        assert trainer.config is not None
        assert trainer.config["model"] == "yolov8n.pt"

    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        trainer = YOLOTrainer()
        assert isinstance(trainer, YOLOTrainer)
        assert isinstance(trainer.config, dict)
