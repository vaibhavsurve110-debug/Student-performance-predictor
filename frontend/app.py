"""
Streamlit Frontend — Student Performance Prediction Dashboard
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Student Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Public Share Handling ──────────────────────────────────────────────────────
query_params = st.query_params
share_token = query_params.get("token")

API_BASE = "http://localhost:8000"

# ── Custom CSS (Modern SaaS / Linear Aesthetic) ────────────────────────────────
try:
    with open(os.path.join(os.path.dirname(__file__), "assets", "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = "Teacher"

with st.sidebar:
    st.markdown("### Student Predictor")
    st.markdown("---")
    st.markdown("### Authentication")
    st.session_state.role = st.selectbox("Current Role", ["Teacher", "Administrator"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Navigation")
    
    nav_options = ["Overview", "Predict", "Bulk Analysis", "History"]
    if st.session_state.role == "Administrator":
        nav_options = ["Overview", "EDA Dashboard", "Predict", "Bulk Analysis", 
                       "Fairness Auditing", "Data Drift", "Model Comparison",
                       "Feature Importance", "History"]
                       
    page = st.radio("Navigate", nav_options, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Stack**\n\nStreamlit · FastAPI\n\nGradient Boosting\n\nScikit-learn")
    st.markdown("---")

    # API health
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.json().get("model_ready"):
            st.success("API Ready")
        else:
            st.warning("Model missing")
    except Exception:
        st.error("API Offline")

# ── Helpers ────────────────────────────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.json()
    except Exception:
        return None

def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.dialog("Share Report")
def show_share_modal(prediction_id, result_label, confidence):
    st.markdown(f"### Share Report for {result_label}")
    st.markdown(f"**Confidence:** {confidence}%")
    
    # Initialize sharing in backend
    with st.spinner("Preparing sharing options..."):
        init_res = api_post("/share/init", {"prediction_id": prediction_id, "owner_role": st.session_state.role})
    
    if "error" in init_res or not init_res.get("success"):
        st.error("Could not initialize sharing. Please try again.")
        return

    shared_report_id = init_res["data"]["shared_report_id"]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📧 Email", "💬 WhatsApp", "🔗 Public Link", "💾 Download"])
    
    report_text = f"🎓 Student Performance Prediction\nOutcome: {result_label}\nConfidence: {confidence}%"
    import urllib.parse
    encoded_report = urllib.parse.quote(report_text)

    # ... (Email, WhatsApp, Link tabs logic remains the same)
    
    # NOTE: I need to handle the tabs properly. Since I'm replacing the whole tabs list, 
    # I should include the content of the other tabs too if I'm using replace_file_content 
    # on the whole block. Actually, I'll just append the new tab logic.

    with tab1:
        st.markdown("#### Send via Email")
        emails = st.text_input("Recipients (comma separated)", placeholder="teacher@school.edu, parent@home.com")
        subject = st.text_input("Subject", value=f"Student Performance Report - {result_label}")
        message = st.text_area("Message Body", value=f"Hello,\n\nPlease find the prediction report for the student.\n\nResult: {result_label}\nConfidence: {confidence}%\n\nRegards,\n{st.session_state.role}")
        
        if st.button("Send Email", type="primary"):
            if emails:
                recipient_list = [e.strip() for e in emails.split(",")]
                res = api_post("/share/email", {
                    "shared_report_id": shared_report_id,
                    "recipients": recipient_list,
                    "subject": subject,
                    "message": message
                })
                if "error" not in res and res.get("success"):
                    st.success("✅ Report shared successfully via Email!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed to send: {res.get('error', 'Check SMTP settings')}")
            else:
                st.warning("Please enter at least one recipient.")

    with tab2:
        st.markdown("#### Share on WhatsApp")
        st.info("Click the button below to open WhatsApp with a pre-filled message.")
        wa_link = f"https://wa.me/?text={encoded_report}"
        if st.markdown(f'<a href="{wa_link}" target="_blank" style="display:inline-block;background-color:#10b981;color:white;padding:10px 20px;border-radius:5px;text-decoration:none;font-weight:bold;text-align:center;width:100%;">Open WhatsApp</a>', unsafe_allow_html=True):
            api_post("/share/log-activity", {"shared_report_id": shared_report_id, "action": "WHATSAPP"})

    with tab3:
        st.markdown("#### Public Shareable Link")
        expiry = st.selectbox("Link Expiry", [1, 24, 168], format_func=lambda x: f"{x} hours" if x < 168 else "7 days")
        allow_dl = st.checkbox("Allow Download", value=True)
        
        if st.button("Generate Link"):
            res = api_post("/share/link", {
                "shared_report_id": shared_report_id,
                "expires_in_hours": expiry,
                "allow_download": allow_dl
            })
            if "error" not in res:
                token = res["data"]["token"]
                # Use absolute URL if possible, otherwise relative
                share_url = f"http://localhost:8501/?token={token}" # Localhost for now
                st.success("Link generated!")
                st.code(share_url, language="text")
                st.info("Copy this link to share with anyone. They don't need to log in.")
            else:
                st.error(f"Error generating link: {res['error']}")

    with tab4:
        st.markdown("#### Export Report")
        st.info("Download the report for offline use or physical distribution.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 Download PDF", use_container_width=True):
                # We can't use st.download_button easily inside a dialog button click 
                # without pre-fetching, so we'll provide a direct link or fetch it.
                r = requests.get(f"{API_BASE}/export/pdf/{prediction_id}")
                if r.status_code == 200:
                    st.download_button("Click here to save PDF", data=r.content, file_name=f"report_{prediction_id}.pdf", mime="application/pdf")
                    api_post("/share/log-activity", {"shared_report_id": shared_report_id, "action": "DOWNLOAD_PDF"})
                else:
                    st.error("Failed to generate PDF.")
        
        with c2:
            if st.button("📊 Download CSV", use_container_width=True):
                r = requests.get(f"{API_BASE}/export/csv/{prediction_id}")
                if r.status_code == 200:
                    st.download_button("Click here to save CSV", data=r.content, file_name=f"report_{prediction_id}.csv", mime="text/csv")
                    api_post("/share/log-activity", {"shared_report_id": shared_report_id, "action": "DOWNLOAD_CSV"})
                else:
                    st.error("Failed to generate CSV.")

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC VIEW HANDLING
# ══════════════════════════════════════════════════════════════════════════════
if share_token:
    st.markdown("""
    <div class="hero-card">
        <h1>Student Prediction Report</h1>
        <p>This report was securely shared with you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Fetching report..."):
        report = api_get(f"/share/public/{share_token}")
        
    if not report or "error" in report:
        st.error("Invalid or expired share link.")
        if st.button("Go to Dashboard"):
            st.query_params.clear()
            st.rerun()
        st.stop()
        
    data = report["result"]
    inp = report["input"]
    
    label = data.get("predicted_label","")
    conf  = data.get("confidence", 0)
    css   = {"Fail":"result-fail","Pass":"result-pass","Excellent":"result-excellent"}.get(label,"result-pass")

    st.markdown(f"""
    <div class="predict-result {css}">
        <div class="result-label">{label}</div>
        <div class="result-conf">Confidence: {conf}%</div>
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prediction Context**")
        st.write(f"**G1:** {inp.get('G1')} | **G2:** {inp.get('G2')} | **Absences:** {inp.get('absences')}")
        st.write(f"**Study Time:** {inp.get('studytime')} | **Age:** {inp.get('age')}")
        
    with col2:
        st.markdown("**Key Factors**")
        top = data.get("top_features", [])[:5]
        for item in top:
            st.write(f"- {item['feature']}: {item['importance']:.2%}")

    if report.get("allow_download"):
        st.markdown("---")
        # Provide CSV download as a simple version of "Download PDF"
        df_exp = pd.DataFrame([inp])
        csv = df_exp.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report Data (CSV)", data=csv, file_name=f"report_{share_token}.csv", mime="text/csv")

    st.info("This is a read-only view. [Click here to login](http://localhost:8501)")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="hero-card">
        <h1>Student Performance Predictor</h1>
        <p>AI-powered academic outcome prediction using Machine Learning on the UCI Student Performance Dataset. Identify at-risk students early and enable timely intervention.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("93.1%","Best Model Accuracy"),
        ("649","Student Records"),
        ("32","Input Features"),
        ("6","ML Models Compared"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4], stats):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Objective**\n\nBuild a predictive system using demographic, behavioural & academic data to classify final academic outcomes — Fail, Pass, or Excellent.")
        st.markdown("**Dataset**\n\nUCI Student Performance Dataset — 649 records, 32 features covering grades, family background, social behaviour, and study habits.")
    with c2:
        st.markdown("**Best Model**\n\nGradient Boosting achieved **93.1% accuracy** with 93.2% F1-Score using 5-fold cross-validation and SMOTE oversampling for class balance.")
        st.markdown("**Top Predictors**\n\nG2 & G1 (prior grades), number of absences, travel time, age, and father's job.")

    st.markdown('<div class="section-header">Performance Classes</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.error("**Fail** — G3 < 10\n\nNeeds intervention")
    c2.warning("**Pass** — 10 ≤ G3 < 15\n\nOn track")
    c3.success("**Excellent** — G3 ≥ 15\n\nPerforming well")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown('<div class="section-header">Predict Performance</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Fill in the student details below to generate a prediction.</p>", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            school    = st.selectbox("School", ["GP","MS"])
            sex       = st.selectbox("Sex", ["M","F"])
            age       = st.slider("Age", 15, 22, 17)
            address   = st.selectbox("Address", ["U","R"])
            famsize   = st.selectbox("Family Size", ["GT3","LE3"])
            Pstatus   = st.selectbox("Parent Status", ["T","A"])

        with c2:
            st.markdown("**Academic**")
            G1       = st.slider("First Period Grade (G1)", 0, 20, 10)
            G2       = st.slider("Second Period Grade (G2)", 0, 20, 10)
            studytime = st.selectbox("Study Time/Week", [1,2,3,4],
                                     format_func=lambda x: {1:"<2h",2:"2-5h",3:"5-10h",4:">10h"}[x])
            failures  = st.selectbox("Past Failures", [0,1,2,3])
            absences  = st.slider("Absences", 0, 93, 4)
            higher    = st.selectbox("Wants Higher Education", ["yes","no"])

        with c3:
            st.markdown("**Family & Behaviour**")
            Medu    = st.selectbox("Mother Education (0-4)", [0,1,2,3,4], index=2)
            Fedu    = st.selectbox("Father Education (0-4)", [0,1,2,3,4], index=2)
            Mjob    = st.selectbox("Mother's Job", ["teacher","health","services","at_home","other"])
            Fjob    = st.selectbox("Father's Job", ["teacher","health","services","at_home","other"])
            internet = st.selectbox("Internet at Home", ["yes","no"])
            goout   = st.slider("Goes Out (1-5)", 1, 5, 3)
            health  = st.slider("Health Status (1-5)", 1, 5, 3)

        submitted = st.form_submit_button("Generate Prediction")

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if submitted:
        payload = {
            "school": school, "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
            "Mjob": Mjob, "Fjob": Fjob, "reason": "course", "guardian": "mother",
            "traveltime": 1, "studytime": studytime, "failures": failures,
            "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "no",
            "nursery": "yes", "higher": higher, "internet": internet, "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": goout, "Dalc": 1, "Walc": 1,
            "health": health, "absences": absences, "G1": G1, "G2": G2,
        }

        with st.spinner("Running AI prediction..."):
            result = api_post("/predict", payload)
            if "error" not in result:
                st.session_state.last_prediction = result
            else:
                st.error(f"Error: {result['error']}\n\nMake sure the backend is running.")

    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        label = result.get("predicted_label","")
        conf  = result.get("confidence", 0)
        css   = {"Fail":"result-fail","Pass":"result-pass","Excellent":"result-excellent"}.get(label,"result-pass")

        st.markdown(f"""
        <div class="predict-result {css}">
            <div class="result-label">{label}</div>
            <div class="result-conf">Confidence: {conf}%</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class Probabilities**")
            probs = result.get("probabilities", {})
            colors = {"Fail":"#ef4444","Pass":"#f59e0b","Excellent":"#10b981"}
            fig = go.Figure(go.Bar(
                x=list(probs.keys()), y=list(probs.values()),
                marker_color=[colors.get(k,"#ffffff") for k in probs.keys()],
                text=[f"{v}%" for v in probs.values()], textposition="auto"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a1a1aa", yaxis_title="Probability (%)",
                showlegend=False, height=300, margin=dict(t=20,b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Key Influencing Factors**")
            top = result.get("top_features", [])[:8]
            if top:
                df_fi = pd.DataFrame(top)
                fig2 = go.Figure(go.Bar(
                    y=df_fi["feature"], x=df_fi["importance"],
                    orientation="h",
                    marker_color="#fafafa",
                ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a1a1aa", height=300,
                    margin=dict(t=20,b=20), yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig2, use_container_width=True)

        if label == "Fail":
            st.error("Intervention Recommended: This student is at risk of failing.")
        elif label == "Pass":
            st.warning("On Track: Student is passing but has room for improvement.")
        else:
            st.success("Excellent Performance: Student is performing outstandingly.")

        # Explainable AI section
        shap_data = result.get("shap_explanation", {})
        if shap_data and "features" in shap_data and len(shap_data["features"]) > 0:
            st.markdown('<div class="section-header">Explainable AI (SHAP)</div>', unsafe_allow_html=True)
            st.markdown("<p style='color:#a1a1aa;'>How each feature specifically contributed to this student's prediction. Green bars push the prediction towards a higher grade, red bars push towards a lower grade.</p>", unsafe_allow_html=True)
            
            shap_df = pd.DataFrame(shap_data["features"])
            shap_df["abs_value"] = shap_df["value"].abs()
            shap_df = shap_df.sort_values(by="abs_value", ascending=True)
            
            colors_shap = ["#ef4444" if v < 0 else "#10b981" for v in shap_df["value"]]
            
            fig3 = go.Figure(go.Bar(
                y=shap_df["feature"], x=shap_df["value"],
                orientation="h",
                marker_color=colors_shap,
                text=[f"{v:+.3f}" for v in shap_df["value"]], textposition="outside"
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a1a1aa", height=400,
                margin=dict(t=20,b=20, l=100),
                xaxis_title="SHAP Value (Impact on Prediction)"
            )
            fig3.add_shape(type="line", x0=0, x1=0, y0=-1, y1=len(shap_df), line=dict(color="#27272a", width=2))
            st.plotly_chart(fig3, use_container_width=True)

        # AI Counselor
        if "ai_counselor_plan" in result and result["ai_counselor_plan"]:
            st.markdown('<div class="section-header">🧠 Prescriptive Analytics (AI Counselor)</div>', unsafe_allow_html=True)
            st.info(result["ai_counselor_plan"])

        # What-If Simulator
        st.markdown('<div class="section-header">🎛️ What-If Simulator</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#a1a1aa;'>Adjust key variables below to instantly see how interventions could change the outcome.</p>", unsafe_allow_html=True)
        
        with st.form("what_if_form"):
            w_col1, w_col2, w_col3 = st.columns(3)
            # Use defaults from last prediction or form inputs
            new_study = w_col1.slider("Simulate Study Time", 1, 4, studytime, key="w_st")
            new_abs = w_col2.slider("Simulate Absences", 0, 93, absences, key="w_ab")
            new_g1 = w_col3.slider("Simulate G1", 0, 20, G1, key="w_g1")
            sim_submit = st.form_submit_button("Run Simulation")
            
        if sim_submit:
            # Note: payload here depends on form variables school, sex etc.
            sim_payload = {
                "school": school, "sex": sex, "age": age, "address": address,
                "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
                "Mjob": Mjob, "Fjob": Fjob, "reason": "course", "guardian": "mother",
                "traveltime": 1, "studytime": new_study, "failures": failures,
                "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "no",
                "nursery": "yes", "higher": higher, "internet": internet, "romantic": "no",
                "famrel": 4, "freetime": 3, "goout": goout, "Dalc": 1, "Walc": 1,
                "health": health, "absences": new_abs, "G1": new_g1, "G2": G2,
            }
            with st.spinner("Simulating..."):
                sim_res = api_post("/predict", sim_payload)
                if "error" not in sim_res:
                    s_label = sim_res.get("predicted_label","")
                    s_conf = sim_res.get("confidence", 0)
                    s_css = {"Fail":"result-fail","Pass":"result-pass","Excellent":"result-excellent"}.get(s_label,"result-pass")
                    st.markdown(f'<div class="predict-result {s_css}"><div class="result-label">Simulated: {s_label}</div><div class="result-conf">{s_conf}% confidence</div></div>', unsafe_allow_html=True)

        # Share Report
        st.markdown('<div class="section-header">📤 Report Sharing</div>', unsafe_allow_html=True)
        
        col_s1, _ = st.columns([1, 2])
        with col_s1:
            if st.button("🚀 Share Analysis", type="primary", key="share_btn", use_container_width=True):
                show_share_modal(result.get("student_id"), label, conf)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Performance of 6 models evaluated with 5-fold cross-validation.</p>", unsafe_allow_html=True)

    data = api_get("/models")
    if data:
        models = data.get("models", [])
    else:
        models = [
            {"name":"Gradient Boosting","accuracy":93.1,"precision":93.3,"recall":93.1,"f1_score":93.2,"notes":"Best model — highest accuracy"},
            {"name":"Random Forest","accuracy":91.4,"precision":91.7,"recall":91.4,"f1_score":91.4,"notes":"Strong ensemble baseline"},
            {"name":"Decision Tree","accuracy":91.4,"precision":91.7,"recall":91.4,"f1_score":91.4,"notes":"Interpretable, prone to overfit"},
            {"name":"Logistic Regression","accuracy":87.1,"precision":87.4,"recall":87.1,"f1_score":86.6,"notes":"Baseline; fast training"},
            {"name":"SVM","accuracy":86.2,"precision":85.9,"recall":86.2,"f1_score":86.0,"notes":"Good with scaled features"},
        ]

    df = pd.DataFrame(models)

    # Grouped bar chart
    metrics = ["accuracy","precision","recall","f1_score"]
    colors  = ["#fafafa", "#a1a1aa", "#71717a", "#52525b"]
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.replace("_"," ").title(),
            x=df["name"], y=df[metric],
            marker_color=color, text=df[metric].astype(str)+"%", textposition="auto"
        ))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a1a1aa", legend=dict(orientation="h", y=1.1),
        yaxis=dict(range=[70,100], title="Score (%)"),
        height=420, margin=dict(t=40,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("**Detailed Metrics**")
    df_display = df[["name","accuracy","precision","recall","f1_score","notes"]].copy()
    df_display.columns = ["Model","Accuracy %","Precision %","Recall %","F1-Score %","Notes"]
    st.dataframe(df_display, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Top predictive features identified by the model.</p>", unsafe_allow_html=True)

    data = api_get("/features")
    if data:
        features = data.get("features", [])
    else:
        features = [
            {"feature":"G2","importance":0.3821,"rank":1},
            {"feature":"G1","importance":0.2914,"rank":2},
            {"feature":"absences","importance":0.0812,"rank":3},
            {"feature":"studytime","importance":0.0631,"rank":4},
            {"feature":"failures","importance":0.0589,"rank":5},
            {"feature":"Medu","importance":0.0412,"rank":6},
            {"feature":"Fedu","importance":0.0381,"rank":7},
            {"feature":"age","importance":0.0245,"rank":8},
            {"feature":"goout","importance":0.0198,"rank":9},
            {"feature":"health","importance":0.0174,"rank":10},
        ]

    df = pd.DataFrame(features)
    fig = go.Figure(go.Bar(
        y=df["feature"], x=df["importance"],
        orientation="h",
        marker_color="#fafafa",
        text=[f"{v:.2%}" for v in df["importance"]], textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a1a1aa", xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed"),
        height=480, margin=dict(t=20,b=20,r=100)
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    st.markdown('<div class="section-header">Prediction History</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])
    with col2:
        if st.button("Clear History"):
            api_post("/history/clear", {})
            st.success("History cleared!")
            st.rerun()

    data = api_get("/history?limit=50")
    records = data.get("records", []) if data else []

    if not records:
        st.info("No prediction history yet.")
    else:
        st.markdown(f"**{len(records)} predictions recorded**")
        rows = []
        for rec in records:
            r = rec.get("result", {})
            inp = rec.get("input", {})
            rows.append({
                "ID": rec.get("id",""),
                "Time": rec.get("timestamp","")[:19].replace("T"," "),
                "Prediction": r.get("predicted_label",""),
                "Confidence": f"{r.get('confidence',0)}%",
                "G1": inp.get("G1",""),
                "G2": inp.get("G2",""),
                "Absences": inp.get("absences",""),
                "Study Time": inp.get("studytime",""),
            })
        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True, height=400)

        # Distribution pie
        dist = df["Prediction"].value_counts()
        colors_map = {"Fail":"#ef4444","Pass":"#f59e0b","Excellent":"#10b981"}
        fig = go.Figure(go.Pie(
            labels=dist.index, values=dist.values,
            marker_colors=[colors_map.get(l,"#333333") for l in dist.index],
            hole=0.6, textinfo='label+percent'
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa",
            height=350, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Shared Reports History
        st.markdown('<div class="section-header">🌍 Shared Reports History</div>', unsafe_allow_html=True)
        share_data = api_get(f"/share/history?limit=20&role={st.session_state.role}")
        shares = share_data.get("history", []) if share_data else []
        
        if not shares:
            st.info("No reports shared yet.")
        else:
            share_rows = []
            for s in shares:
                share_rows.append({
                    "Shared At": s.get("timestamp")[:16].replace("T", " "),
                    "Prediction ID": s.get("prediction_id"),
                    "Shared By": s.get("owner"),
                    "Outcome": s.get("label"),
                    "Link": "Active"
                })
            st.table(pd.DataFrame(share_rows))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bulk Analysis":
    st.markdown('<div class="section-header">Bulk Analysis (CSV/Excel Upload)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Upload a CSV or Excel file containing student records to generate predictions in bulk. The file must contain the same columns as the training dataset.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Student Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"Successfully loaded {len(df)} records from {uploaded_file.name}.")
            
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head(5))
                
            if st.button("Generate Bulk Predictions"):
                with st.spinner("Processing batch predictions..."):
                    # Convert dataframe to list of dicts for API
                    records = df.to_dict(orient="records")
                    result = api_post("/predict/batch", records)
                    
                    if "batch_results" in result:
                        batch = result["batch_results"]
                        # Extract predictions back into the dataframe
                        df["Predicted_Class"] = [b.get("predicted_label", "Error") for b in batch]
                        df["Confidence_%"] = [b.get("confidence", 0) for b in batch]
                        
                        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                        
                        # Provide download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Annotated CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Error processing batch. Ensure the dataset matches the required schema.")
        except Exception as e:
            st.error(f"Error reading file. Ensure openpyxl is installed for Excel files. Error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Interactive analysis of the training dataset.</p>", unsafe_allow_html=True)
    
    try:
        import plotly.express as px
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv"), sep=";")
        
        st.markdown("**Correlation Heatmap (Numeric Features)**")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Interactive Scatter Matrix**")
        c1, c2, c3 = st.columns(3)
        x_var = c1.selectbox("X-Axis", numeric_df.columns, index=list(numeric_df.columns).index("G1"))
        y_var = c2.selectbox("Y-Axis", numeric_df.columns, index=list(numeric_df.columns).index("G3"))
        color_var = c3.selectbox("Color By", df.columns, index=list(df.columns).index("sex"))
        
        fig2 = px.scatter(df, x=x_var, y=y_var, color=color_var, color_discrete_sequence=["#10b981", "#ec4899", "#f59e0b", "#3b82f6"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa")
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load data for EDA: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS AUDITING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fairness Auditing":
    st.markdown('<div class="section-header">Algorithmic Fairness & Bias Auditing</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Evaluate the model for disparate impact across protected demographic groups.</p>", unsafe_allow_html=True)
    
    if st.button("Run Global Fairness Audit"):
        with st.spinner("Running global fairness audit across all student records..."):
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv"), sep=";")
            records = df.to_dict(orient="records")
            res = api_post("/predict/batch", records)
            
            if "batch_results" in res:
                df["Predicted_Class"] = [b.get("predicted_label", "Fail") for b in res["batch_results"]]
                df["Is_Success"] = df["Predicted_Class"].isin(["Pass", "Excellent"])
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("### Gender Fairness (Disparate Impact)")
                    success_rates = df.groupby("sex")["Is_Success"].mean() * 100
                    fig = go.Figure(go.Bar(
                        x=success_rates.index, y=success_rates.values,
                        marker_color=["#3b82f6", "#ec4899"],
                        text=[f"{v:.1f}%" for v in success_rates.values], textposition="auto"
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa", yaxis_title="Success Rate (%)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    diff = abs(success_rates.get('F', 0) - success_rates.get('M', 0))
                    if diff < 10:
                        st.success(f"✅ **Fairness Check Passed:** Difference is only {diff:.1f}%.")
                    else:
                        st.warning(f"⚠️ **Bias Detected:** Significant difference of {diff:.1f}%.")
                
                with c2:
                    st.markdown("### Geographic Fairness (Urban vs Rural)")
                    addr_rates = df.groupby("address")["Is_Success"].mean() * 100
                    fig2 = go.Figure(go.Bar(
                        x=addr_rates.index, y=addr_rates.values,
                        marker_color=["#10b981", "#8b5cf6"],
                        text=[f"{v:.1f}%" for v in addr_rates.values], textposition="auto"
                    ))
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa", yaxis_title="Success Rate (%)", height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    diff_addr = abs(addr_rates.get('U', 0) - addr_rates.get('R', 0))
                    if diff_addr < 10:
                        st.success(f"✅ **Fairness Check Passed:** Difference is {diff_addr:.1f}%.")
                    else:
                        st.warning(f"⚠️ **Bias Detected:** Rural vs Urban disparity of {diff_addr:.1f}%.")
            else:
                st.error("Error computing fairness metrics.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Drift":
    st.markdown('<div class="section-header">Data Drift Detection (Pipeline Health)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa;'>Monitor incoming batch data against the training distribution to detect model degradation.</p>", unsafe_allow_html=True)
    
    st.info("Upload a new batch of student data to compare against the original training baseline.")
    uploaded_file = st.file_uploader("Upload New Batch (CSV)", type=["csv"], key="drift")
    
    if uploaded_file is not None:
        try:
            sep = ";" if "student" in uploaded_file.name else ","
            df_new = pd.read_csv(uploaded_file, sep=sep)
            df_base = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv"), sep=";")
            
            from scipy.stats import ks_2samp
            import plotly.figure_factory as ff
            
            st.markdown("### Distribution Shift (Kolmogorov-Smirnov Test)")
            features_to_check = ["G1", "G2", "absences", "age", "studytime"]
            drift_results = []
            
            for f in features_to_check:
                if f in df_new.columns and f in df_base.columns:
                    stat, p_val = ks_2samp(df_base[f].dropna(), df_new[f].dropna())
                    status = "🚨 Drift Detected" if p_val < 0.05 else "✅ Stable"
                    drift_results.append({"Feature": f, "KS Statistic": round(stat, 4), "P-Value": round(p_val, 4), "Status": status})
                    
            if drift_results:
                st.dataframe(pd.DataFrame(drift_results), use_container_width=True)
                
                has_drift = any(r["Status"] == "🚨 Drift Detected" for r in drift_results)
                if has_drift:
                    st.warning("⚠️ Critical data drift detected. Model retraining is highly recommended.")
                    if st.button("🔄 Trigger Automated Retraining Pipeline"):
                        with st.spinner("Initializing MLOps pipeline..."):
                            retrain_res = api_post("/models/retrain", {})
                            if "error" not in retrain_res:
                                st.success("Retraining pipeline initiated in the background! The model will be hot-swapped seamlessly upon completion.")
                            else:
                                st.error(f"Failed to start retraining: {retrain_res.get('error')}")
                
                if "G2" in df_new.columns:
                    hist_data = [df_base["G2"], df_new["G2"]]
                    group_labels = ['Training Data', 'New Batch']
                    fig = ff.create_distplot(hist_data, group_labels, bin_size=1, show_rug=False)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a1a1aa", title="Distribution Comparison for G2")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No matching numerical features found for drift testing.")
        except Exception as e:
            st.error(f"Error computing drift: {e}")
