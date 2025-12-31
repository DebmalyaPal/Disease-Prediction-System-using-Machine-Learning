class DiseaseEnsemble:
    """
    Ensemble model wrapper for disease prediction.

    This class combines predictions from three different classifiers
    (Gaussian Naive Bayes, Random Forest, and SVM) to produce a more
    stable and reliable probability distribution over possible diseases.
    Each model is expected to implement `predict_proba`.

    Args:
        gnb: Trained Gaussian Naive Bayes classifier.
        rf: Trained Random Forest classifier.
        svm: Trained Support Vector Machine classifier with probability=True.
        disease_id_name_map (dict): Mapping of class indices to disease names.
    """

    def __init__(self, gnb, rf, svm, disease_id_name_map):
        self.gnb = gnb
        self.rf = rf
        self.svm = svm
        self.disease_id_name_map = disease_id_name_map

    def predict_top3(self, X):
        """
        Predict the top 3 most likely diseases for a given symptom vector.
        
        This method:
        - Accepts a pandas DataFrame containing a single row of symptom features.
        - Computes class probabilities from all three ensemble models
        (GaussianNB, RandomForest, SVM).
        - Averages the probability distributions to form an ensemble output.
        - Extracts the top 3 highest probability disease classes.
        - Maps class indices to human readable disease names.
        - Returns probabilities as percentages (0 - 100).

        Args:
            X (pandas.DataFrame):
                A DataFrame of shape (1, n_features) representing the encoded
                symptom vector. Column order must match the model's training data.

        Returns:
            list[dict]: A list of three dictionaries, each containing:
                - "disease_id": Integer class index.
                - "disease": Human-readable disease name.
                - "probability": Ensemble probability (%) rounded to 2 decimals.

            Example:
            [
                {"disease_id": 5, "disease": "Diabetes", "probability": 72.14},
                {"disease_id": 12, "disease": "Hypertension", "probability": 18.77},
                {"disease_id": 3, "disease": "Migraine", "probability": 4.92}
            ]
        """

        # Ensure all models return probabilities
        p1 = self.gnb.predict_proba(X)
        p2 = self.rf.predict_proba(X)
        p3 = self.svm.predict_proba(X)

        # Combine probabilities
        combined = (p1 + p2 + p3) / 3

        # Get top 3 indices
        idx = combined[0].argsort()[::-1][:3]
        probs = combined[0][idx]

        # Map to disease names
        return [
            {
                "disease_id": i,
                "disease": self.disease_id_name_map[i].title(), 
                "probability": round(float(probs[j] * 100), 2)
            } for j, i in enumerate(idx)
        ]
