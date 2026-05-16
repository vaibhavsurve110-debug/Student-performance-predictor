import os
import pandas as pd
from fpdf import FPDF
from datetime import datetime

class ExportEngine:
    @staticmethod
    def to_csv(input_data: dict, result_data: dict) -> str:
        """Generate CSV string for the report."""
        combined = {**input_data, **result_data}
        # Remove complex objects
        combined.pop("probabilities", None)
        combined.pop("top_features", None)
        combined.pop("shap_explanation", None)
        
        df = pd.DataFrame([combined])
        return df.to_csv(index=False)

    @staticmethod
    def to_pdf(input_data: dict, result_data: dict) -> bytes:
        """Generate PDF bytes for the report."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Student Performance Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)

        # Result Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Prediction Result", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(40, 10, "Outcome:", 0)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 10, str(result_data.get("predicted_label")), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(40, 10, "Confidence:", 0)
        pdf.cell(0, 10, f"{result_data.get('confidence')}%", ln=True)
        pdf.ln(5)

        # Top Factors
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Key Influencing Factors", ln=True)
        pdf.set_font("Arial", "", 10)
        for idx, feat in enumerate(result_data.get("top_features", [])[:5]):
            pdf.cell(0, 8, f"{idx+1}. {feat['feature']}: {feat['importance']:.2%}", ln=True)
        pdf.ln(5)

        # Input Context
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Student Context", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Grid layout for inputs
        inputs = list(input_data.items())[:12] # Show first 12 for brevity
        for i in range(0, len(inputs), 2):
            k1, v1 = inputs[i]
            line = f"{k1}: {v1}"
            if i + 1 < len(inputs):
                k2, v2 = inputs[i+1]
                line += f"  |  {k2}: {v2}"
            pdf.cell(0, 8, line, ln=True)

        return pdf.output()
