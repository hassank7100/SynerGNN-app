# streamlit_app.py
import json
import streamlit as st
import torch, itertools, json, pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

CHECKPOINT = "gnn_synergy_model.pth"
METADATA   = "drugs.json"
TOP_N      = 10   # how many combos to list in “rank my inventory”

# ──────────────────────────────────────────────────────────
# 1.  Load model + predictor + drug metadata (with caching)
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GNN…")
def load_model():
    # embeddings you saved earlier
    z = torch.load("node_embeddings.pt", map_location="cpu")

    # load real names + smiles
    with open("drugs.json") as fp:
        drug_meta = json.load(fp)           # {int: {"name":..., "smiles":...}}

    class LinkPred(torch.nn.Module):
        def forward(self, z, edges):
            u, v = z[edges[0]], z[edges[1]]
            return (u * v).sum(dim=1)

    return z, LinkPred(), drug_meta



    encoder = GCNEncoder(in_dim, hidden, out_dim)
    decoder = LinkPred()
    encoder.load_state_dict(ckpt["model_state_dict"])
    decoder.load_state_dict(ckpt["predictor_state_dict"])
    encoder.eval(); decoder.eval()

    # fake graph to compute embeddings once (no gradients needed)
    num_nodes = len(ckpt["idx_map"])
    dummy_x   = torch.eye(num_nodes, in_dim) * 0  # placeholder
    dummy_edge = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        z = encoder(dummy_x, dummy_edge)

    return z, decoder, ckpt["idx_map"]

@st.cache_resource
def load_drug_meta():
    with open(METADATA) as fp:
        return {int(k): v for k, v in json.load(fp).items()}

embeddings, predictor, idx2smiles = load_model()
drug_meta = load_drug_meta()
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
        prob = predict_pair(choiceA, choiceB)
        nameA = drug_meta[choiceA]["name"]
        nameB = drug_meta[choiceB]["name"]
        label = "Synergistic ✅" if prob > 0.5 else "Not synergistic ❌"
        st.metric(f"{nameA} + {nameB}", f"{prob:.3f}", label)

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
            probs   = [predict_pair(i, j) for i, j in combos]
            rows = [{
                "Drug A": drug_meta[i]["name"],
                "Drug B": drug_meta[j]["name"],
                "Predicted Synergy": round(p, 3)
            } for (i, j), p in zip(combos, probs)]
            
            df = pd.DataFrame(rows).sort_values("Predicted Synergy", ascending=False).reset_index(drop=True)
            top_pairs = df.head(TOP_N)
            
            st.write(f"### 🏆 Top {TOP_N} Predicted Synergistic Pairs")
            
            # Add label arrows based on synergy probability
            def label_arrow(prob):
                return "🔼 Likely Synergy" if prob > 0.5 else "🔽 Unlikely Synergy"
            
            top_pairs["Prediction"] = top_pairs["Predicted Synergy"].apply(label_arrow)
            
            # Prettier table with green/red gradient
            st.dataframe(
                top_pairs.style.background_gradient(
                    subset=["Predicted Synergy"], cmap="RdYlGn"  # RED ➔ YELLOW ➔ GREEN
                ).format({
                    "Predicted Synergy": "{:.3f}"
                }),
                use_container_width=True
            )


st.caption("Model trained on real CRKP synergy data • probabilities >0.5 suggest likely synergy")
