import streamlit as st
import itertools
import torch
import numpy as np
from load_artifacts import (
    load_prediction_artifacts,   # <-- your helper from earlier
    predict_synergy              # <-- your helper from earlier
)

# ---------- Load model & artifacts ----------
MODEL_PATH = "gnn_synergy_model.pth"
model, predictor, smiles_map, idx_map = load_prediction_artifacts(MODEL_PATH)

# You must also load data.x and train_data.edge_index exactly as in training
# For deployment, pickle or torch.save them into a file and load here:
import pickle, os
with open("data_artifacts.pkl", "rb") as fh:
    data_x, train_edge_index, drug_names = pickle.load(fh)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="SynerGNN – Antibiotic Synergy", layout="centered")
st.title("🦠 SynerGNN – Predicting Synergistic Antibiotic Pairs")

st.markdown(
"""
**How it works**  
*Select two drugs manually* **or** *pick the inventory you have* and let the model suggest
the best combinations (highest predicted probability of synergy).  
The current model was trained on **28 antibiotics** and 38 real interactions.
"""
)

# ----- Manual Pair Prediction -----
st.header("1  Manual prediction")
col1, col2 = st.columns(2)

drug_a = col1.selectbox("Drug 1", list(drug_names))
drug_b = col2.selectbox("Drug 2", list(drug_names), index=1)

if st.button("Predict this pair"):
    if drug_a == drug_b:
        st.warning("Select two *different* drugs.")
    else:
        idx1, idx2 = drug_names.index(drug_a), drug_names.index(drug_b)
        prob = predict_synergy(model, predictor, data_x, train_edge_index,
                               idx1, idx2)
        st.write(f"**Predicted synergy probability:** {prob:.2%}")
        st.success("Synergistic ✅" if prob > 0.5 else "Not synergistic ❌")

st.divider()

# ----- Inventory-Based Ranking -----
st.header("2  Top-ranked combinations from your inventory")

inventory = st.multiselect(
    "Select the antibiotics you have in stock",
    drug_names,            # full list
    default=["Colistin", "Meropenem", "Tigecycline"]
)

top_n = st.slider("Number of suggestions (N)", 1, 10, 5, 1)

if st.button("Show best combinations"):
    if len(inventory) < 2:
        st.warning("Pick at least two drugs.")
    else:
        # Generate all unordered pairs
        pairs = list(itertools.combinations(inventory, 2))
        results = []
        for d1, d2 in pairs:
            i1, i2 = drug_names.index(d1), drug_names.index(d2)
            prob = predict_synergy(model, predictor, data_x, train_edge_index,
                                   i1, i2)
            results.append((d1, d2, prob))
        # Sort high → low
        results.sort(key=lambda x: x[2], reverse=True)
        st.subheader(f"Top {top_n} predicted synergistic pairs")
        for d1, d2, p in results[:top_n]:
            st.write(f"**{d1} + {d2}** → {p:.2%}")

st.caption("Model trained on limited, imbalanced data – probabilities are exploratory only.")
