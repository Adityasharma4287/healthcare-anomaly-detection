"""
api/app.py
Flask application factory and entry point.
"""

import os
import sys
import logging
from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "dashboard", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "dashboard", "static"),
    )

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    app.config["JSON_SORT_KEYS"] = False

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    from api.routes import api_bp
    app.register_blueprint(api_bp)

    # Dashboard route
    @app.route("/")
    def dashboard():
        return render_template("index.html")

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {"error": "Internal server error"}, 500

    logger.info("Flask application created.")
    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
