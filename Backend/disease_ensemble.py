class DiseaseEnsemble:
    def __init__(self, gnb, rf, svm, disease_id_name_map):
        self.gnb = gnb
        self.rf = rf
        self.svm = svm
        self.disease_id_name_map = disease_id_name_map

    def predict_top3(self, X):
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
