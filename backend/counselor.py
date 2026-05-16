"""
counselor.py — Prescriptive Analytics engine using LLMs (Gemini).
"""

import os

def generate_intervention_plan(shap_features, predicted_label):
    if predicted_label == "Excellent":
        return "Student is performing exceptionally well. Ensure they remain challenged with advanced coursework or leadership roles."
        
    api_key = os.getenv("GOOGLE_API_KEY")
    feature_summary = ", ".join([f"{f['feature']} (impact: {f['value']:.2f})" for f in shap_features[:3]])
    prompt = f"A student is predicted to {predicted_label}. The key driving factors are: {feature_summary}. Provide a short 3-sentence actionable intervention plan for a teacher."
    
    if not api_key:
        # Fallback rule-based
        plan = "Based on the AI data: "
        for f in shap_features[:2]:
            if f['value'] < 0 and f['feature'] == 'absences':
                plan += "Monitor absences closely and schedule a meeting with parents to resolve attendance blocks. "
            elif f['value'] < 0 and f['feature'] == 'studytime':
                plan += "Recommend the student attends mandatory after-school study groups to increase study volume. "
            elif f['value'] < 0 and f['feature'] in ['G1', 'G2']:
                plan += "Prior grades are pulling the trajectory down; assign remedial tutoring for foundational concepts. "
        if plan == "Based on the AI data: ":
            plan += "Schedule a 1-on-1 meeting to discuss academic progress and identify hidden blockers."
        return plan + " (Note: Export GOOGLE_API_KEY to enable dynamic Gemini AI generation)."
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate AI advice: {e}"
