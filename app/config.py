import os

class Config:
    """
    Base configuration for the AI Note-Taking App.
    Contains default settings and environment variables for all environments.
    """
    # Flask settings
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")

    # SQLAlchemy / Database configuration
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

    # Paths for your saved inference model & tokenizers
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

    # Sequence length parameters (must match how you trained your model)
    MAX_LENGTH_INPUT = int(os.environ.get("MAX_LENGTH_INPUT", 50))
    MAX_LENGTH_TARGET = int(os.environ.get("MAX_LENGTH_TARGET", 20))

    

class DevelopmentConfig(Config):
    """
    Development configuration.
    """
    DEBUG = True


class TestingConfig(Config):
    """
    Testing configuration.
    """
    TESTING = True


class ProductionConfig(Config):
    """
    Production configuration.
    """
    DEBUG = False
