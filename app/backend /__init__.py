# app/__init__.py

import os
import logging
from flask import Flask
from .config import DevelopmentConfig  # or ProductionConfig/TestingConfig as needed
from .blueprints.notes import notes_bp

def create_app(config_object=DevelopmentConfig):
    """
    Application factory to create and configure the Flask app.
    """
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Configure logging to file + console at desired level
    log_level = getattr(logging, app.config.get('LOGGING_LEVEL', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    # Register Blueprints
    app.register_blueprint(notes_bp, url_prefix='/api/notes')

    # Health-check endpoint example
    @app.route('/health', methods=['GET'])
    def health_check():
        return {'status': 'OK'}, 200

    return app
