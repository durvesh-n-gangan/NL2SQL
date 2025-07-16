import streamlit as st
from main import handle_query, get_saved_questions

# Page configuration
st.set_page_config(page_title="Gemini SQL Assistant", page_icon="💡")

# Main title
st.title("💬 Gemini‑Powered SQL Assistant")

# --- Sidebar: Enhanced History Display ---
st.sidebar.title("🕘 Past Questions")
past_qs = get_saved_questions()

if past_qs:
    # Reverse the order so most recent questions appear first
    for i, q in enumerate(reversed(past_qs), 1):
        if st.sidebar.button(f"{i}. {q}", key=f"hist_{i}"):
            st.session_state['selected_query'] = q
else:
    st.sidebar.info("No past questions yet.")

# --- Main Input ---
query = st.session_state.get('selected_query', "")
query = st.text_input("Ask a question about your data:", value=query)

if st.button("Submit") and query.strip():
    with st.spinner("Processing..."):
        answer = handle_query(query)
        st.markdown("### ✅ Answer")
        st.write(answer)
        
        # Clear selected after submitting
        if 'selected_query' in st.session_state:
            del st.session_state['selected_query']