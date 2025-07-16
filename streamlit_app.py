import streamlit as st
from main import handle_query, get_saved_questions

st.set_page_config(page_title="Gemini SQL Assistant", page_icon="💡")

st.title("💬 Gemini‑Powered SQL Assistant")
st.sidebar.title("🕘 Past Questions")
past_qs = get_saved_questions()

if past_qs:
    for i, q in enumerate(reversed(past_qs), 1):
        if st.sidebar.button(f"{i}. {q}", key=f"hist_{i}"):
            st.session_state['selected_query'] = q
else:
    st.sidebar.info("No past questions yet.")

query = st.session_state.get('selected_query', "")
query = st.text_input("Ask a question about your data:", value=query)

if st.button("Submit") and query.strip():
    with st.spinner("Processing..."):
        answer = handle_query(query)
        st.markdown("### ✅ Answer")
        st.write(answer)
        
        if 'selected_query' in st.session_state:
            del st.session_state['selected_query']