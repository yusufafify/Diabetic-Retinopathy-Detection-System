from datetime import datetime

class KnowledgeBase:
    def __init__(self):
        self.recommendations = {
            0: "No action required. Regular eye checkups every 1-2 years.",
            1: "Mild Non-Proliferative DR detected. Recommend regular monitoring every 6-12 months. Consider lifestyle modifications.",
            2: "Moderate Non-Proliferative DR. Recommend closer monitoring. Lifestyle changes and possible interventions.",
            3: "Severe Non-Proliferative DR. Immediate medical intervention may be required. Recommend referral to an ophthalmologist.",
            4: "Proliferative DR. Immediate referral to a specialist. Possible treatment options: laser therapy or surgery."
        }

    def get_recommendation(self, grade: int):
        """Return recommendation based on DR grade."""
        return self.recommendations.get(grade, "Unknown Grade. Consultation required.")

    def get_features(self, segmentation_class: int):
        """Simulate feature extraction based on the segmentation class."""
        # Placeholder for feature extraction logic based on segmentation result
        features = [
            {"name": "Microaneurysms", "present": int(segmentation_class == 1), "severity": "Mild" if segmentation_class == 1 else "Severe"},
            {"name": "Hemorrhages", "present": int(segmentation_class >= 2), "severity": "Moderate" if segmentation_class >= 2 else "None"},
            {"name": "Exudates", "present": int(segmentation_class >= 3), "severity": "Severe" if segmentation_class == 3 else "Mild"},
            {"name": "Cotton Wool Spots", "present": int(segmentation_class == 4), "severity": "Moderate" if segmentation_class == 4 else "None"},
            {"name": "Vessel Abnormalities", "present": int(segmentation_class >= 2), "severity": "Severe" if segmentation_class == 4 else "Mild"},
            {"name": "Macular Edema", "present": int(segmentation_class >= 3), "severity": "Severe" if segmentation_class == 3 else "Mild"},
            {"name": "Neovascularization", "present": int(segmentation_class == 4), "severity": "Severe" if segmentation_class == 4 else "None"},
            {"name": "Retinal Detachment", "present": int(segmentation_class == 4), "severity": "Critical" if segmentation_class == 4 else "None"}
        ]
        return features
