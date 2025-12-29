import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
from marshmallow import Schema, fields, ValidationError
import pandas as pd
import joblib

# ---------------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
swagger = Swagger(app)

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
    disease_lookup = {d["name"].lower(): d for d in disease_info}

    for pred in predictions:
        name = pred["disease"].lower()
        extra = disease_lookup.get(name, {})
        enriched.append({
            "disease": pred["disease"],
            "probability": pred["probability"],
            "description": extra.get("description", ""),
            "precautions": extra.get("precautions", []),
            "severity": extra.get("severity", "")
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
        SYMPTOM_INFO = load_json_file("./Data/Symptom_Info.json")

        logger.info("Resources loaded successfully.")

    except Exception as e:
        logger.critical(f"Failed to initialize resources: {e}")
        raise


# ---------------------------------------------------------
# API Request Data Validation Schema
# ---------------------------------------------------------
class SymptomSchema(Schema):
    symptoms = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        required=True
    )

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
    --------
    JSON response containing enriched predictions.
    ---
    tags:
        -   Prediction
    consumes:
        -   application/json
    parameters:
        -   in: body
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
                                        type: number
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
    try:
        json_data = request.get_json() 

        if not json_data: 
            return jsonify({"error": "Invalid or missing JSON body"}), 400 
        
        # Validate and deserialize 
        data = symptom_schema.load(json_data) 
        symptom_dict = data["symptoms"]

        # Convert to DataFrame
        df_input = build_input_dataframe(symptom_dict)

        # Run prediction
        predictions = MODEL.predict_top3(df_input)

        # Enrich with disease metadata
        enriched_output = enrich_predictions(predictions, DISEASE_INFO)

        return jsonify({
            "success": True,
            "predictions": enriched_output
        })
    
    except ValidationError as ve: 
        # Input validation error (400) 
        logger.warning(f"Validation error: {ve.messages}") 
        return jsonify({"error": "Validation error", "details": ve.messages}), 400 
    except Exception as e: 
        logger.error(f"Prediction error: {e}") 
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
