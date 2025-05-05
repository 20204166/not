import os

class Config:
    """
    Base configuration for the AI Note-Taking App.

    Contains default settings and environment variables for all environments.
    """
    # Flask settings
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")

    # SQLAlchemy / Database configuration
    # Default to MySQL 'taskdb' on localhost; override via DATABASE_URI env var
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URI",
        "mysql+pymysql://root:password@localhost:3306/taskdb"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # External API key for integrations
    EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "")

    # Logging configuration
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
    LOG_FILE = os.environ.get("LOG_FILE", "app.log")

    # Paths for model, tokenizers, and training data
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        "app/models/saved_model/summarization_model.keras"
    )
    TOKENIZER_INPUT_PATH = os.environ.get(
        "TOKENIZER_INPUT_PATH",
        "app/models/saved_model/tokenizer_input.json"
    )
    TOKENIZER_TARGET_PATH = os.environ.get(
        "TOKENIZER_TARGET_PATH",
        "app/models/saved_model/tokenizer_target.json"
    )
    TRAINING_DATA_PATH = os.environ.get(
        "TRAINING_DATA_PATH",
        "app/models/data/text/training_data.json"
    )

    # Sequence length parameters (must match training configuration)
    MAX_LENGTH_INPUT = int(os.environ.get("MAX_LENGTH_INPUT", 50))
    MAX_LENGTH_TARGET = int(os.environ.get("MAX_LENGTH_TARGET", 20))

    # Special token indices (must match tokenizer configuration)
    START_TOKEN_INDEX = int(os.environ.get("START_TOKEN_INDEX", 1))
    END_TOKEN_INDEX = int(os.environ.get("END_TOKEN_INDEX", 2))

    # Training parameters (used if you retrain or evaluate)
    TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", 30))
    TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE", 16))
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 50))

    # Flags for model rebuild and device usage
    FORCE_REBUILD_MODEL = os.environ.get(
        "FORCE_REBUILD_MODEL", "False"
    ).lower() in ("true", "1", "yes")
    USE_CPU_FOR_TRAINING = os.environ.get(
        "USE_CPU_FOR_TRAINING", "False"
    ).lower() in ("true", "1", "yes")


class DevelopmentConfig(Config):
    """
    Development configuration.
    """
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DEV_DATABASE_URI",
        "sqlite:///dev_app.db"
    )
    LOGGING_LEVEL = os.environ.get("DEV_LOGGING_LEVEL", "DEBUG")
    EXTERNAL_API_KEY = os.environ.get("DEV_EXTERNAL_API_KEY", "")


class TestingConfig(Config):
    """
    Testing configuration.
    """
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "TEST_DATABASE_URI",
        "sqlite:///:memory:"
    )
    LOGGING_LEVEL = os.environ.get("TEST_LOGGING_LEVEL", "DEBUG")
    EXTERNAL_API_KEY = os.environ.get("TEST_EXTERNAL_API_KEY", "")


class ProductionConfig(Config):
    """
    Production configuration.
    """
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URI",
        "postgresql://user:password@hostname/dbname"
    )
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "WARNING")
    EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "")
