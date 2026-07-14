import os
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "FactTrack"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./facttrack.db")

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAINED_MODELS_DIR: str = os.path.join(BASE_DIR, "trained_models")
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")

    CATEGORY_MODEL_PATH: str = os.path.join(TRAINED_MODELS_DIR, "category_model")
    BIAS_MODEL_PATH: str = os.path.join(TRAINED_MODELS_DIR, "bias_model")

    # Fallback pre-trained model for zero-shot classification when fine-tuned weights are missing
    ZERO_SHOT_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # Model parameters
    MAX_SEQUENCE_LENGTH: int = 512

    # Training Hyperparameters
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    BATCH_SIZE: int = 8
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
