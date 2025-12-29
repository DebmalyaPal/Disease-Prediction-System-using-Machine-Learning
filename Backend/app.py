import json
import logging
import time 
import uuid
from flask import Flask, request, jsonify, g, has_request_context
from flask_cors import CORS
from flask_limiter import Limiter 
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from marshmallow import Schema, fields, ValidationError
from werkzeug.exceptions import HTTPException
import pandas as pd
import joblib

from logging_config import configure_logger
from disease_ensemble import DiseaseEnsemble
from custom_exceptions import UnknownSymptomException


# ---------------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
swagger = Swagger(app)
app.config["RATELIMIT_ENABLED"] = True
app.url_map.strict_slashes = False


# ---------------------------------------------------------
# API Rate Limit
# ---------------------------------------------------------
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logger = configure_logger()


# ---------------------------------------------------------
# Global Variables (Loaded Once at Startup)
# ---------------------------------------------------------
MODEL = None
DISEASE_INFO = None
SYMPTOM_INFO = None


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_json_file(path: str):
    """
    Load a JSON file safely.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON is invalid.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {path}")
        raise


def load_model(path: str):
    """
    Load a machine learning model using joblib.

    Parameters
    ----------
    path : str
        Path to the model file.

    Returns
    -------
    object
        Loaded model object.
    """
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def build_input_dataframe(symptom_dict: dict) -> pd.DataFrame:
    """
    Convert incoming symptom dictionary into a pandas DataFrame.

    Parameters
    ----------
    symptom_dict : dict
        Dictionary of symptom_id → 0/1 or values.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame suitable for model prediction.
    """
    try:
        df = pd.DataFrame([symptom_dict])
        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        raise


def enrich_predictions(predictions: list, disease_info: list):
    """
    Enrich model predictions with additional disease metadata.

    Parameters
    ----------
    predictions : list
        List of dicts returned by the model (disease + probability).
    disease_info : list
        List of disease metadata dictionaries.

    Returns
    -------
    list
        Enriched list of prediction dictionaries.
    """
    enriched = []
    disease_lookup = {d["id"]: d for d in disease_info}

    for prediction in predictions:
        id = prediction['disease_id']
        
        extra = disease_lookup.get(id, {})

        description = extra.get("description", "")
        precautions = [ 
            extra.get("precaution1", ""), 
            extra.get("precaution2", ""), 
            extra.get("precaution3", ""), 
            extra.get("precaution4", "") 
        ]
        
        enriched.append({
            "id": str(id),
            "name": prediction["disease"].title(),
            "description": description.capitalize(),
            "precautions": [ precaution.capitalize() for precaution in precautions ],
            "probability": f"{prediction['probability']} %"
        })

    return enriched


# ---------------------------------------------------------
# Load Resources at Startup
# ---------------------------------------------------------
@app.before_first_request
def load_resources():
    """
    Load model and JSON data once when the Flask app starts.
    """
    global MODEL, DISEASE_INFO, SYMPTOM_INFO

    try:
        logger.info("Loading model and JSON data...")

        MODEL = load_model("./Model/ensemble_pipeline.pkl")
        DISEASE_INFO = load_json_file("./Data/Disease_Info.json")
        SYMPTOM_INFO = load_json_file("./Data/Symptoms_Info.json")

        logger.info("Resources loaded successfully.")

    except Exception as e:
        logger.critical(f"Failed to initialize resources: {e}")
        raise


# ---------------------------------------------------------
# Correlation IDs of Requests & Request/Response Logging Middleware
# ---------------------------------------------------------
@app.before_request
def add_correlation_id():
    g.correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))


@app.before_request
def log_request():
    g.start_time = time.time()
    logger.info(f"[REQUEST] {request.method} {request.path} - Body: {request.get_json(silent=True)}")


@app.after_request
def log_response(response):
    start = getattr(g, "start_time", None)
    if start is not None:
        duration = round((time.time() - start) * 1000, 2)
        logger.info(
            f"[RESPONSE] {request.method} {request.path} - "
            f"Status: {response.status_code} - Duration: {duration}ms"
        )
    else:
        # No start_time → request was blocked early (rate limit, static, debugger)
        logger.info(
            f"[RESPONSE] {request.method} {request.path} - "
            f"Status: {response.status_code} - Duration: N/A"
        )

    return response


# ---------------------------------------------------------
# API Request Data Validation Schema
# ---------------------------------------------------------
class SymptomSchema(Schema):
    symptoms = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        required=True
    )

class DiseaseInfoSchema(Schema):
    name = fields.String(required=True)
    description = fields.String(required=True)
    precautions = fields.List(fields.String(), required=True)
    severity = fields.String(required=True)

disease_info_schema = DiseaseInfoSchema(many=True)
symptom_schema = SymptomSchema()

# ---------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict top 3 probable diseases based on symptom input.

    Expected JSON Input:
        {
            "symptoms": {
                "fever": 1,
                "cough": 0,
                "fatigue": 1
            }
        }

    Returns: 
        JSON response containing enriched predictions.
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            symptoms:
              type: object
              additionalProperties:
                type: integer
              example:
                fever: 1
                cough: 0
                fatigue: 1
    responses:
      200:
        description: Successful prediction
        schema:
          type: object
          properties:
            success:
              type: boolean
            predictions:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: number
                  name: 
                    type: string
                  probability:
                    type: string
                  description:
                    type: string
                  precautions:
                    type: array
                    items:
                      type: string
      400:
        description: Bad request (missing or invalid data)
      500:
        description: Internal server error
    """
    response, status_code = None, None
    
    try:
        json_data = request.get_json() 

        if not json_data: 
            response = {
                "success": False, 
                "error": { 
                    "type": "BadRequest", 
                    "message": "Invalid or missing JSON body." 
                }
            }
            status_code = 400 
        else:
            # Validate and deserialize 
            data = symptom_schema.load(json_data) 
            request_symptom_dict = data["symptoms"]
            
            all_symptom_dict = { symptom["code"]: 0 for symptom in SYMPTOM_INFO } 

            for request_symptom, is_present in request_symptom_dict.items():
                if request_symptom in all_symptom_dict:
                    all_symptom_dict[request_symptom] = is_present
                else:
                    raise UnknownSymptomException(f"Unknown symptom code: {request_symptom}")

            # Convert to DataFrame
            df_input = build_input_dataframe(all_symptom_dict)

            # Run prediction
            predictions = MODEL.predict_top3(df_input)

            # Enrich with disease metadata
            enriched_output = enrich_predictions(predictions, DISEASE_INFO)

            response = {
                "success": True,
                "predictions": enriched_output
            }
            status_code = 200
        
    except ValidationError as ve: 
        # Input validation error (400) 
        logger.warning(f"Validation error: {ve.messages}") 
        response = {
            "success": False, 
            "error": { 
                "type": "ValidationError", 
                "message": "Invalid request payload."
            } 
        }
        status_code = 400

    finally:
        return jsonify(response), status_code


# ---------------------------------------------------------
# Global Error Handler
# ---------------------------------------------------------
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """
    Handles all HTTP errors (400, 404, 405, etc.)
    Ensures consistent JSON output.
    """
    logger.warning(f"HTTPException: {e.code} - {e.description}")

    return jsonify({
        "success": False,
        "error": {
            "type": e.__class__.__name__,
            "message": e.description
        }
    }), e.code


@app.errorhandler(UnknownSymptomException)
def handle_unknown_symptom(e):
    logger.warning(f"UnknownSymptomException: {str(e)}")

    return jsonify({
        "success": False,
        "error": {
            "type": "UnknownSymptomException",
            "message": str(e)
        }
    }), 400


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    """
    Handles all unexpected server errors.
    """
    logger.error(f"Unhandled Exception: {str(e)}")

    return jsonify({
        "success": False,
        "error": {
            "type": "InternalServerError",
            "message": "An unexpected error occurred. Please try again later."
        }
    }), 500


# ---------------------------------------------------------
# Ready & Health API endpoint
# ---------------------------------------------------------
@app.route("/ready")
def ready():
    if MODEL is None or DISEASE_INFO is None or SYMPTOM_INFO is None:
        return jsonify({"ready": False}), 503
    return jsonify({"ready": True}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "Ok",
        "service": "disease-prediction-api",
        "timestamp": time.time()
    }), 200


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

