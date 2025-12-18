import io
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw

# Optional imports for advanced features
try:
    import shap  # type: ignore
except ImportError:
    shap = None

try:
    import plotly.graph_objects as go # type: ignore
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- Configuration ---
APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
MODEL_PATH = PROJECT_ROOT / "clintox_xgboost_final.pkl"
METADATA_PATH = PROJECT_ROOT / "model_metadata.json"
STYLE_PATH = APP_ROOT / "style.css"

FP_BITS = 2048
FP_RADIUS = 2

st.set_page_config(
    page_title="ClinTox toxicity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Utils ---
def inject_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body{font-family: 'Inter', sans-serif;}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model() -> Tuple[object, dict]:
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return model, metadata

def fingerprint_from_smiles(smiles: str) -> Optional[np.ndarray]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

def molfile_to_smiles(file_bytes: bytes, filename: str) -> Optional[str]:
    name = filename.lower()
    try:
        if name.endswith(".sdf"):
            suppl = Chem.ForwardSDMolSupplier(io.BytesIO(file_bytes))
            mols = [m for m in suppl if m]
            return Chem.MolToSmiles(mols[0]) if mols else None
        if any(name.endswith(ext) for ext in [".mol", ".mol2", ".pdb"]):
            mol = Chem.MolFromMolBlock(file_bytes.decode(errors="ignore"), sanitize=True)
            return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None
    return None

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(400, 300))

@st.cache_data
def load_reference_data():
    """Load reference dataset for similarity search."""
    try:
        df = pd.read_csv(PROJECT_ROOT / "data/clintox.csv")
        # Pre-calculate fingerprints for speed
        mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
        valid_idxs = [i for i, m in enumerate(mols) if m is not None]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=2048) for i in valid_idxs]
        return df.iloc[valid_idxs].reset_index(drop=True), fps
    except Exception:
        return None, None

def find_similar_molecules(query_mol, df_ref, fps_ref, top_n=3):
    """Find most similar molecules in the reference dataset."""
    if df_ref is None or fps_ref is None:
        return []
        
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps_ref)
    
    # Get top N indices
    top_indices = np.argsort(sims)[-top_n:][::-1]
    results = []
    for idx in top_indices:
        row = df_ref.iloc[idx]
        results.append({
            "SMILES": row["SMILES"],
            "Similarity": sims[idx],
            "FDA_APPROVED": row["FDA_APPROVED"],
            "CT_TOX": row["CT_TOX"]
        })
    return results

def check_lipinski(mol):
    """Check Lipinski's Rule of 5."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    rules = {
        "MW < 500 Da": (mw < 500, f"{mw:.1f}"),
        "LogP < 5": (logp < 5, f"{logp:.2f}"),
        "H-Bond Donors < 5": (hbd < 5, f"{hbd}"),
        "H-Bond Acceptors < 10": (hba < 10, f"{hba}")
    }
    passed = sum([r[0] for r in rules.values()])
    return rules, passed

def predict_single(model, smiles: str) -> Optional[dict]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Calculate Descriptors
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'Num_C': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']),
            'Num_N': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N']),
            'Num_O': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O']),
            'Num_Halogens': len([a for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I']]),
            'Num_Chiral_Centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'Num_Aromatic_Atoms': len([a for a in mol.GetAtoms() if a.GetIsAromatic()]),
            'Total_Formal_Charge': Chem.GetFormalCharge(mol)
        }
        
        # Calculate Fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
        fp_arr = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)
        
        # Create DataFrame with correct column order
        desc_df = pd.DataFrame([descriptors])
        fp_df = pd.DataFrame([fp_arr], columns=[f"FP_{i}" for i in range(FP_BITS)])
        
        # Combine: Descriptors + Fingerprints
        X = pd.concat([desc_df, fp_df], axis=1)
        
        proba = float(model.predict_proba(X)[:, 1][0])
        return {"prob": proba, "features": X}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def plot_gauge(probability: float, key: str = "gauge"):
    if not PLOTLY_AVAILABLE:
        st.progress(probability)
        st.caption(f"Toxicity Probability: {probability:.1%}")
        return

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Toxicity Probability (%)", 'font': {'size': 16, 'color': '#6c757d'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#dc3545" if probability > 0.5 else "#28a745"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#dee2e6",
            'steps': [
                {'range': [0, 50], 'color': "#e8f5e9"},
                {'range': [50, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#212529", 'family': "Inter"},
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# --- Main App ---
inject_css()
model, metadata = load_model()

# Initialize Session State
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'current_smiles' not in st.session_state:
    st.session_state.current_smiles = ""

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🧬 ClinTox AI")
    page = st.radio("Navigation", ["Single Analysis", "Batch Prediction", "About Model"])
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.8em; color:gray;'>
        <b>Model Info:</b><br>
        Type: XGBoost (ECFP4)<br>
        ROC-AUC: {metadata.get('test_rocauc', '0.95+')}<br>
        Training Data: {metadata.get('n_training_samples', '1400+')}
    </div>
    """, unsafe_allow_html=True)

# --- Page: Single Analysis ---
if page == "Single Analysis":
    st.title("🧪 Single Molecule Analysis")
    st.markdown("Analyze individual compounds for clinical toxicity.")
    
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.subheader("1. Input Molecule")
        
        input_tab1, input_tab2 = st.tabs(["✏️ Manual SMILES", "📂 Upload File"])
        
        # Helper to update state
        def update_smiles(smiles):
            st.session_state.current_smiles = smiles
            st.session_state.prediction_result = None

        with input_tab1:
            def manual_callback():
                if st.session_state.manual_input:
                    update_smiles(st.session_state.manual_input)
            
            st.text_input("Paste SMILES string", placeholder="C1=CC=CC=C1", key="manual_input", on_change=manual_callback)

        with input_tab2:
            uploaded_file = st.file_uploader("Upload .csv, .sdf, .mol, .pdb", type=["csv", "sdf", "mol", "pdb"])
            if uploaded_file:
                filename = uploaded_file.name.lower()
                
                # Handle CSV (Batch Mode inside Tab)
                if filename.endswith(".csv"):
                    try:
                        df_batch = pd.read_csv(uploaded_file)
                        if "SMILES" in df_batch.columns:
                            st.info(f"📄 CSV loaded with {len(df_batch)} compounds.")
                            
                            if st.button("Predict & Generate Report", key="btn_batch_tab"):
                                with st.spinner("Processing batch..."):
                                    results = []
                                    progress_bar = st.progress(0)
                                    for i, smi in enumerate(df_batch["SMILES"]):
                                        res = predict_single(model, str(smi))
                                        if res:
                                            results.append({
                                                "SMILES": smi,
                                                "Probability": res["prob"],
                                                "Prediction": "Toxic" if res["prob"] > 0.5 else "Non-Toxic"
                                            })
                                        else:
                                            results.append({"SMILES": smi, "Probability": None, "Prediction": "Error"})
                                        progress_bar.progress((i + 1) / len(df_batch))
                                    
                                    res_df = pd.DataFrame(results)
                                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                                    st.success("✅ Processing complete!")
                                    st.download_button(
                                        label="📥 Download Predictions",
                                        data=csv_data,
                                        file_name="clintox_predictions.csv",
                                        mime="text/csv"
                                    )
                        else:
                            st.error("CSV must contain a 'SMILES' column.")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")

                # Handle Molecular Files (Single Analysis)
                else:
                    uploaded_file.seek(0)
                    parsed_smiles = molfile_to_smiles(uploaded_file.read(), uploaded_file.name)
                    if parsed_smiles:
                        update_smiles(parsed_smiles)
                    else:
                        st.error("Could not parse file.")

        # Quick Examples
        st.markdown("#### Quick Examples")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        if ex_col1.button("Aspirin"): update_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        if ex_col2.button("Caffeine"): update_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        if ex_col3.button("Digitoxin"): update_smiles("CC12CCC(CC1CCC3C2CCC4(C3CCC4C5=CC(=O)OC5)O)OC6CC(C(C(O6)O)O)OC7CC(C(C(O7)O)O)OC8CC(C(C(O8)O)O)O")

    with right_col:
        st.subheader("2. Analysis & Prediction")
        
        if st.session_state.current_smiles:
            # Visualization
            img = draw_molecule(st.session_state.current_smiles)
            if img:
                st.image(img, caption="Molecular Structure", use_container_width=False, width=350)
            else:
                st.warning("Invalid SMILES structure")
            
            # Prediction Button
            st.markdown("#### Prediction Results")
            if st.button("Analyze Toxicity", type="primary", use_container_width=True):
                with st.spinner("Analyzing molecular fingerprints..."):
                    time.sleep(0.5)
                    result = predict_single(model, st.session_state.current_smiles)
                    st.session_state.prediction_result = result
            
            # Display Results from Session State
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                prob = result["prob"]
                is_toxic = prob > 0.5
                
                # Result Card
                card_color = "var(--danger)" if is_toxic else "var(--success)"
                bg_color = "#ffebee" if is_toxic else "#e8f5e9"
                label = "TOXIC" if is_toxic else "NON-TOXIC"
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border-left: 6px solid {card_color};
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                ">
                    <h2 style="margin:0; color:{card_color}; font-size: 1.8rem;">{label}</h2>
                    <p style="margin:5px 0 0 0; color: #495057; font-weight: 500;">Confidence Score: {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                plot_gauge(prob, key="single_gauge")

                # Radar Chart for Descriptors
                if PLOTLY_AVAILABLE and "features" in result:
                    with st.expander("🕸️ Physicochemical Profile (Radar Chart)"):
                        feats = result["features"].iloc[0]
                        radar_cols = ['MW', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
                        r_vals = [
                            min(feats['MW']/500, 1), 
                            min(max(feats['LogP']/5, 0), 1), 
                            min(feats['TPSA']/150, 1), 
                            min(feats['NumHDonors']/5, 1), 
                            min(feats['NumHAcceptors']/10, 1)
                        ]
                        
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=r_vals,
                            theta=radar_cols,
                            fill='toself',
                            name='Molecule'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=False,
                            height=300,
                            margin=dict(l=40, r=40, t=20, b=20)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

                # --- Similarity Search ---
                with st.expander("🔍 Similar Known Compounds"):
                    with st.spinner("Searching database..."):
                        mol = Chem.MolFromSmiles(st.session_state.current_smiles)
                        if mol:
                            df_ref, fps_ref = load_reference_data()
                            if df_ref is not None:
                                sim_mols = find_similar_molecules(mol, df_ref, fps_ref)
                                if sim_mols:
                                    sim_cols = st.columns(3)
                                    for i, sim in enumerate(sim_mols):
                                        with sim_cols[i]:
                                            sim_img = draw_molecule(sim["SMILES"])
                                            st.image(sim_img, use_container_width=True)
                                            st.caption(f"Sim: {sim['Similarity']:.2f}")
                                            status = "Toxic" if sim["CT_TOX"] == 1 else "Safe"
                                            fda = "FDA Approved" if sim["FDA_APPROVED"] == 1 else "Not FDA"
                                            st.markdown(f"**{status}** | {fda}")
                            else:
                                st.error("Could not load reference data.")
                        else:
                            st.error("Invalid molecule structure.")
                
                # SHAP Explanation (if available)
                if shap and "features" in result:
                    with st.expander("📊 Key Structural Features (SHAP)"):
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(result["features"])
                            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                            
                            if np.sum(np.abs(vals)) > 0:
                                st.caption("Positive values push towards Toxicity, negative towards Safety.")
                                feature_importance = pd.DataFrame({
                                    'Feature': [f"Bit {i}" for i in range(FP_BITS)],
                                    'Impact': vals[0]
                                }).sort_values(by='Impact', key=abs, ascending=False).head(10)
                                st.bar_chart(feature_importance.set_index('Feature'))
                        except Exception:
                            pass
        else:
            st.info("👈 Please select or enter a molecule to begin analysis.")

# --- Page: Batch Prediction ---
elif page == "Batch Prediction":
    st.title("📦 Batch Prediction")
    st.write("Upload a CSV file containing a 'SMILES' column to predict toxicity for multiple compounds at once.")
    
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if batch_file:
        try:
            df_batch = pd.read_csv(batch_file)
            if "SMILES" in df_batch.columns:
                st.write(f"Loaded {len(df_batch)} compounds.")
                if st.button("Run Batch Prediction", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, smi in enumerate(df_batch["SMILES"]):
                        res = predict_single(model, str(smi))
                        if res:
                            results.append({
                                "SMILES": smi,
                                "Probability": res["prob"],
                                "Prediction": "Toxic" if res["prob"] > 0.5 else "Non-Toxic"
                            })
                        else:
                            results.append({"SMILES": smi, "Probability": None, "Prediction": "Error"})
                        progress_bar.progress((i + 1) / len(df_batch))
                    
                    res_df = pd.DataFrame(results)
                    st.success("Batch processing complete!")
                    st.dataframe(res_df)
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            else:
                st.error("CSV must contain a 'SMILES' column.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- Page: About ---
elif page == "About Model":
    st.title("ℹ️ About ClinTox AI")
    st.markdown("""
    ### Project Overview
    This application uses a machine learning model to predict the clinical toxicity of chemical compounds based on their molecular structure.
    
    ### Methodology
    *   **Algorithm**: XGBoost Classifier
    *   **Features**: 
        *   Morgan Fingerprints (Radius 2, 2048 bits)
        *   Physicochemical Descriptors (MW, LogP, TPSA, etc.)
    *   **Dataset**: ClinTox (part of MoleculeNet), containing qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.
    
    ### Performance
    The model achieves a ROC-AUC score of approximately **0.95** on the test set, indicating high capability in distinguishing between toxic and non-toxic compounds.
    """)
