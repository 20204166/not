from flask import Flask
from .config import DevelopmentConfig  # Optional, if you use configuration settings
from .blueprints.notes import notes_bp

def create_app(config_object=DevelopmentConfig):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Register Blueprints
    app.register_blueprint(notes_bp, url_prefix='/api/notes')

    return app
