# streamlit_app.py
import json
import streamlit as st
import torch, itertools, pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Constants
CHECKPOINT = "gnn_synergy_model.pth"
TOP_N = 10

# Load model and metadata
@st.cache_resource(show_spinner="Loading GNN…")
def load_model():
    z = torch.load("node_embeddings.pt", map_location="cpu")
    with open("drugs.json") as fp:
        drug_meta = json.load(fp)

    class LinkPred(torch.nn.Module):
        def forward(self, z, edges):
            u, v = z[edges[0]], z[edges[1]]
            return (u * v).sum(dim=1)

    return z, LinkPred(), drug_meta

embeddings, predictor, drug_meta = load_model()
num_drugs = len(drug_meta)

# ──────────────────────────────────────────────────────────
# 2.  Helper to compute probability for any pair (i,j)
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_pair(i, j):
    edge = torch.tensor([[i], [j]])
    logit = predictor(embeddings, edge)[0]
    prob  = torch.sigmoid(logit).item()
    return prob

# ──────────────────────────────────────────────────────────
# 3.  Streamlit UI
# ──────────────────────────────────────────────────────────
st.title("SynerGNN – predict antibiotic synergy for *Klebsiella pneumoniae*")

tab1, tab2 = st.tabs(["🔍 Check one pair", "📋 Rank my inventory"])

# ----- Pair checker UI -----
with tab1:
    st.subheader("Check a single combination")

    # make a list of indices once
    all_indices = sorted(drug_meta)        # e.g. [0,1,2, …]

    # Drop-down for Drug A
    choiceA = st.selectbox(
        "Drug A",
        options=all_indices,
        format_func=lambda i: f"{drug_meta[i]['name']}  (#{i})"
    )

    # Drop-down for Drug B (default to a different item)
    choiceB = st.selectbox(
        "Drug B",
        options=all_indices,
        index=1 if len(all_indices) > 1 else 0,
        format_func=lambda i: f"{drug_meta[i]['name']}  (#{i})"
    )

    if st.button("Predict synergy →", key="pair-btn"):
        # --- prediction ---
        idxA, idxB = int(choiceA), int(choiceB)
        prob  = predict_pair(idxA, idxB)
        nameA = drug_meta[idxA]["name"]
        nameB = drug_meta[idxB]["name"]
    
        # difference from the neutral 0.50 threshold
        delta_val   = prob - 0.50
        verdict_txt = "Synergistic ✅" if prob > 0.50 else "Not synergistic ❌"
    
        # put BOTH pieces in the delta line
        delta_str = f"{delta_val:+.3f} • {verdict_txt}"
    
        st.metric(
            label = f"{nameA}  +  {nameB}",
            value = f"{prob:.3f}",
            delta = delta_str,           # coloured red or green automatically
            help  = "Green ↑ = >0.50 (likely synergy) • Red ↓ = <0.50"
        )



# ---- Inventory ranker
with tab2:
    st.subheader("Rank the best pairs among selected antibiotics")
    inventory = st.multiselect(
        "Select antibiotics you have available",
        options=all_indices,
        format_func=lambda i: drug_meta[i]["name"],
    )

    if len(inventory) < 2:
        st.info("Select at least two drugs.")
    else:
        if st.button(f"Rank top {TOP_N} pairs →", key="rank-btn"):
            combos  = list(itertools.combinations(inventory, 2))
            probs = [predict_pair(int(i), int(j)) for i, j in combos]
            rows    = [{
                "Drug A": drug_meta[i]["name"],
                "Drug B": drug_meta[j]["name"],
                "Predicted Synergy": round(p, 3)
            } for (i, j), p in zip(combos, probs)]
            df = pd.DataFrame(rows).sort_values("Predicted Synergy", ascending=False)
            st.write(f"### Top {TOP_N} predicted synergistic pairs")
            df_sorted = pd.DataFrame(rows).sort_values(
                "Predicted Synergy", ascending=False, ignore_index=True
            )
            
            # choose TOP_N rows
            top_df = df_sorted.head(TOP_N)
            
            # build a red-to-green gradient (RdYlGn reversed = Gn->Rd)
            styler = (
                top_df.style.background_gradient(
                    subset=["Predicted Synergy"], cmap="RdYlGn_r", vmin=0, vmax=1
                )
                .format({"Predicted Synergy": "{:.3f}"})
            )
            
            st.write(f"### 🏆 Top {TOP_N} predicted synergistic pairs")
            st.dataframe(styler, use_container_width=True)


st.caption("Model trained on real CRKP synergy data • probabilities >0.5 suggest likely synergy")
