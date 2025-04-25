# app/__init__.py
import logging
from flask import Flask
from .config import DevelopmentConfig  # adjust as needed
from .blueprints.notes import notes_bp

def create_app(config_object=DevelopmentConfig):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # logging
    log_level = getattr(logging, app.config.get('LOGGING_LEVEL', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    # register blueprints
    app.register_blueprint(notes_bp, url_prefix='/api/notes')

    @app.route('/health')
    def health_check():
        return {'status': 'OK'}, 200

    return app
