# streamlit_app.py
import json
import streamlit as st
import torch, itertools, pandas as pd
# import torch.nn.functional as F # Not needed if using pre-computed embeddings
# from torch_geometric.nn import GCNConv # Not needed

# Constants
# --- UPDATE THESE FILENAMES ---
EMBEDDINGS_FILE = "node_embeddings_combined.pt" # Output of GCN(data.x, data.edge_index)
DRUGS_INFO_FILE = "drugs_combined.json"       # Maps index to drug name and canonical SMILES
# CHECKPOINT_FILE = "gnn_synergy_model_combined.pth" # Not strictly needed if only using embeddings for prediction
# -----------------------------
TOP_N = 10

# Load pre-computed embeddings and metadata
@st.cache_resource(show_spinner="Loading model data…") # Updated spinner text
def load_data():
    try:
        # Load the final node embeddings (output of the GCN encoder)
        embeddings_tensor = torch.load(EMBEDDINGS_FILE, map_location="cpu")
        
        with open(DRUGS_INFO_FILE) as fp:
            drug_meta_loaded = json.load(fp) # Keys are strings like "0", "1"
            # Convert string keys from JSON to integers for easier lookup
            drug_meta_converted = {int(k): v for k, v in drug_meta_loaded.items()}


        # This LinkPred class directly uses the pre-computed node embeddings
        class LinkPred(torch.nn.Module):
            def forward(self, z_embeddings, edges): # z_embeddings is the full embedding matrix
                u_embed = z_embeddings[edges[0]] # Get embedding for drug u
                v_embed = z_embeddings[edges[1]] # Get embedding for drug v
                # Ensure they are correctly shaped for batch dot product if needed
                # For single pair, edges[0] and edges[1] will be scalar indices
                return (u_embed * v_embed).sum(dim=-1) # Sum along the feature dimension

        return embeddings_tensor, LinkPred(), drug_meta_converted
    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please ensure '{EMBEDDINGS_FILE}' and '{DRUGS_INFO_FILE}' are present. Details: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading model data: {e}")
        return None, None, None


embeddings, predictor, drug_meta = load_data()

if embeddings is None or predictor is None or drug_meta is None:
    st.stop() # Stop execution if data loading failed

num_drugs = embeddings.shape[0] # Number of drugs is from the embeddings tensor

# ──────────────────────────────────────────────────────────
# 2.  Helper to compute probability for any pair (i,j)
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_pair(idx_a, idx_b):
    # Ensure indices are integers
    idx_a, idx_b = int(idx_a), int(idx_b)
    
    # Create edge tensor for the pair
    # The LinkPred class expects a batch of edges, even if it's just one.
    # edges should be [2, num_edges_to_predict]
    edge = torch.tensor([[idx_a], [idx_b]], dtype=torch.long) # Shape: [2, 1]

    # The predictor expects the full embedding matrix and the edge indices
    logit = predictor(embeddings, edge) # Pass the full embeddings tensor
    
    # If multiple edges were predicted, logit might be a tensor.
    # For a single pair, it should be a tensor with one element.
    prob  = torch.sigmoid(logit[0]).item() # Get the first (and only) element
    return prob

# ──────────────────────────────────────────────────────────
# 3.  Streamlit UI
# ──────────────────────────────────────────────────────────
st.title("SynerGNN: AI-Driven Prediction of Antibiotic Synergy Against *Klebsiella pneumoniae*")
st.markdown("""
Welcome to SynerGNN! This tool uses a Graph Convolutional Network (GCN) trained on a curated dataset
of experimental antibiotic interactions against *Klebsiella pneumoniae*. 
Select two drugs to predict their likelihood of synergy.
""")

tab1, tab2 = st.tabs(["Check One Pair", "Rank My Inventory"])

# ----- Pair checker UI -----
with tab1:
    st.subheader("Check a Single Combination")

    if not drug_meta: # Handle case where drug_meta might be empty
        st.warning("Drug metadata could not be loaded.")
    else:
        # Create options for selectbox: list of (display_name, index)
        # drug_meta keys are now integers
        drug_options_list = sorted(drug_meta.keys()) 

        choiceA_idx = st.selectbox(
            "Drug A",
            options=drug_options_list,
            format_func=lambda i: f"{drug_meta[i]['name']} (ID: {i})" # Access drug_meta with integer i
        )

        choiceB_idx = st.selectbox(
            "Drug B",
            options=drug_options_list,
            index=min(1, len(drug_options_list)-1) if len(drug_options_list) > 1 else 0, # Safe default index
            format_func=lambda i: f"{drug_meta[i]['name']} (ID: {i})"
        )

        if st.button("Predict synergy →", key="pair-btn"):
            if choiceA_idx == choiceB_idx:
                st.warning("Please select two different drugs.")
            else:
                # --- prediction ---
                prob  = predict_pair(choiceA_idx, choiceB_idx)
                nameA = drug_meta[choiceA_idx]["name"] # Use integer index directly
                nameB = drug_meta[choiceB_idx]["name"]
            
                delta_val   = prob - 0.50
                verdict_txt = "Synergistic ✅" if prob > 0.50 else "Not synergistic / Additive / Indifferent ❌"
            
                delta_str = f"{delta_val:+.3f} • {verdict_txt}"
            
                st.metric(
                    label = f"{nameA}  +  {nameB}",
                    value = f"{prob:.3f}",
                    delta = delta_str,
                    help  = "Higher probability (>0.50, Green ↑) suggests likely synergy. Lower probability (<0.50, Red ↓) suggests non-synergy."
                )
                st.caption(f"SMILES A: `{drug_meta[choiceA_idx]['smiles']}`")
                st.caption(f"SMILES B: `{drug_meta[choiceB_idx]['smiles']}`")


# ---- Inventory ranker
with tab2:
    st.subheader("Rank the Best Pairs Among Selected Antibiotics")
    if not drug_meta:
        st.warning("Drug metadata could not be loaded.")
    else:
        # drug_meta keys are now integers
        inventory_indices = st.multiselect(
            "Select antibiotics you have available (at least 2)",
            options=sorted(drug_meta.keys()), # Use integer keys for options
            format_func=lambda i: f"{drug_meta[i]['name']} (ID: {i})", # Access with integer i
        )

        if len(inventory_indices) < 2:
            st.info("Please select at least two drugs to rank combinations.")
        else:
            if st.button(f"Rank top {TOP_N} pairs →", key="rank-btn"):
                # inventory_indices are already integers
                combos  = list(itertools.combinations(inventory_indices, 2))
                
                with st.spinner(f"Predicting for {len(combos)} combinations..."):
                    rows = []
                    for i, j in combos:
                        p = predict_pair(i, j)
                        rows.append({
                            "Drug A": drug_meta[i]["name"],
                            "Drug B": drug_meta[j]["name"],
                            "Predicted Synergy Probability": p # Keep full precision for sorting
                        })
                
                if not rows:
                    st.warning("No predictions could be made for the selected drugs.")
                else:
                    df_sorted = pd.DataFrame(rows).sort_values(
                        "Predicted Synergy Probability", ascending=False, ignore_index=True
                    )
                    
                    top_df = df_sorted.head(TOP_N)
                    
                    styler = (
                        top_df.style.background_gradient(
                            subset=["Predicted Synergy Probability"], cmap="RdYlGn", vmin=0, vmax=1
                        )
                        .format({"Predicted Synergy Probability": "{:.3f}"})
                        .set_properties(**{'width': '150px'}, subset=['Drug A', 'Drug B'])
                        .set_properties(**{'width': '200px'}, subset=['Predicted Synergy Probability'])
                    )
                    
                    st.write(f"### Top {min(TOP_N, len(top_df))} Predicted Synergistic Pairs")
                    st.dataframe(styler, use_container_width=False) # Set to False for better control with set_properties

st.divider()
st.caption("SynerGNN v1.1 • Model trained on combined real-world *K. pneumoniae* synergy data (122 interactions, 76 drugs). Probabilities >0.50 suggest likely synergy. For research and educational purposes only.")
