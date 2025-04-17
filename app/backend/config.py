import os

class Config:
    """
    Base configuration for the AI Note-Taking App.
    
    This class contains default settings and environment variables that apply to all environments.
    """
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")
    
    # Database configuration:
    DATABASE_URI = os.environ.get("DATABASE_URI", "sqlite:///app.db")
    
    # External API key for integrations.
    EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "your-api-key-here")
    
    # Logging configuration.
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
    LOG_FILE = os.environ.get("LOG_FILE", "app.log")
    
    # Model and Tokenizer paths (for the custom summarization model)
    MODEL_PATH = os.environ.get("MODEL_PATH", "app/models/saved_model/summarization_model.h5")
    TOKENIZER_INPUT_PATH = os.environ.get("TOKENIZER_INPUT_PATH", "app/models/saved_model/tokenizer_input.json")
    TOKENIZER_TARGET_PATH = os.environ.get("TOKENIZER_TARGET_PATH", "app/models/saved_model/tokenizer_target.json")
    
    # Training parameters:
    TRAINING_DATA_FILE = os.environ.get("TRAINING_DATA_FILE", "app/models/data/text/training_data.json")
    TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", 30))
    TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE", 16))
    MAX_LENGTH_INPUT = int(os.environ.get("MAX_LENGTH_INPUT", 50))
    MAX_LENGTH_TARGET = int(os.environ.get("MAX_LENGTH_TARGET", 20))
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 50))
    
    # Flags to control model rebuilding and device usage.
    FORCE_REBUILD_MODEL = os.environ.get("FORCE_REBUILD_MODEL", "False").lower() in ("true", "1", "yes")
    USE_CPU_FOR_TRAINING = os.environ.get("USE_CPU_FOR_TRAINING", "False").lower() in ("true", "1", "yes")
    
    # Additional configurations can be added here as needed.

class DevelopmentConfig(Config):
    """
    Development configuration.
    
    Used during local development and debugging.
    """
    DEBUG = True
    DATABASE_URI = os.environ.get("DEV_DATABASE_URI", "sqlite:///dev_app.db")
    LOGGING_LEVEL = os.environ.get("DEV_LOGGING_LEVEL", "DEBUG")
    EXTERNAL_API_KEY = os.environ.get("DEV_EXTERNAL_API_KEY", "dev-api-key-here")

class TestingConfig(Config):
    """
    Testing configuration.
    
    Used during automated testing to ensure a controlled environment.
    """
    TESTING = True
    DATABASE_URI = os.environ.get("TEST_DATABASE_URI", "sqlite:///:memory:")
    LOGGING_LEVEL = os.environ.get("TEST_LOGGING_LEVEL", "DEBUG")
    EXTERNAL_API_KEY = os.environ.get("TEST_EXTERNAL_API_KEY", "test-api-key-here")

class ProductionConfig(Config):
    """
    Production configuration.
    
    Used when deploying the application to production environments.
    """
    DEBUG = False
    DATABASE_URI = os.environ.get("DATABASE_URI", "postgresql://user:password@hostname/dbname")
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "WARNING")
    EXTERNAL_API_KEY = os.environ.get("EXTERNAL_API_KEY", "prod-api-key-here")
