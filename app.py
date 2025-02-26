"""
This module is the main Streamlit application for the Medical Chatbot.
It provides a user interface for interacting with the chatbot and collecting feedback.
"""
import streamlit as st
import requests
import pandas as pd
from graph import generate_and_display_graphs

# ------------------- CONFIGURATION -------------------
API_URL = "http://127.0.0.1:8000/answer"
FEEDBACK_FILE = "feedback.csv"

# Load feedback data


def load_feedback():
    try:
        return pd.read_csv(FEEDBACK_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Question", "Answer", "Satisfaction"])

# Save feedback data


def save_feedback(question, response, satisfaction):
    feedback_df = load_feedback()
    new_feedback = pd.DataFrame(
        [[question, response, satisfaction]], columns=[
            "Question", "Answer", "Satisfaction"]
    )
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)


# ------------------- STREAMLIT INTERFACE -------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Medical Chatbot")

# Tabs
tabs = st.tabs(["💬 Chatbot", "📊 Feedback & Analysis"])

# ------------------- 💬 CHATBOT -------------------
with tabs[0]:
    st.header("💬 Chat with the Medical AI")
    st.write(
        "Ask a question and receive an instant response with relevant sources and focus area.")

    # Initialize chat history if not present
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display conversation in an interactive chat format
    for idx, chat in enumerate(st.session_state.history[::-1]):
        with st.chat_message("user"):
            st.markdown(f"**🗨️ {chat['question']}**")

        with st.chat_message("assistant"):
            st.markdown(f"**📝 Answer:** {chat['response']}")
            st.markdown(f"🔍 **Source:** {chat['sources']}")
            st.markdown(f"📌 **Focus Area:** {chat['focus_area']}")
            st.markdown(
                f"💡 **Similarity Score:** {chat['similarity']} ({chat['similarity_type']})")

            with st.expander("📊 View Response Metrics"):
                metrics = chat["metrics"]
                if metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        cosine_similarity_value = metrics.get(
                            "cosine_similarity", {}).get("generated_answer", 0)
                        st.metric(label="Cosine Similarity", value=round(
                            float(cosine_similarity_value), 4))
                    with col2:
                        rouge1_value = metrics.get(
                            "rouge_scores", {}).get("rouge1", 0)
                        st.metric(label="ROUGE-1",
                                  value=round(float(rouge1_value), 4))
                    with col3:
                        rouge2_value = metrics.get(
                            "rouge_scores", {}).get("rouge2", 0)
                        st.metric(label="ROUGE-2",
                                  value=round(float(rouge2_value), 4))

                    col4, col5 = st.columns(2)
                    with col4:
                        rougeL_value = metrics.get(
                            "rouge_scores", {}).get("rougeL", 0)
                        st.metric(label="ROUGE-L",
                                  value=round(float(rougeL_value), 4))
                    with col5:
                        response_time_value = metrics.get("response_time", 0)
                        st.metric(label="Response Time (s)", value=round(
                            float(response_time_value), 4))
                else:
                    st.warning("No metrics available for this response.")
            # Feedback buttons
            if "feedback" not in chat:
                chat["feedback"] = None

            if chat["feedback"] is None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Satisfied", key=f"good_{idx}"):
                        chat["feedback"] = "Satisfied"
                        save_feedback(chat["question"],
                                      chat["response"], "Satisfied")
                        st.success("Thank you for your feedback!")

                with col2:
                    if st.button("🤷 Neutral", key=f"neutral_{idx}"):
                        chat["feedback"] = "Neutral"
                        save_feedback(chat["question"],
                                      chat["response"], "Neutral")
                        st.info("Feedback recorded.")

                with col3:
                    if st.button("👎 Dissatisfied", key=f"bad_{idx}"):
                        chat["feedback"] = "Dissatisfied"
                        save_feedback(chat["question"],
                                      chat["response"], "Dissatisfied")
                        st.error(
                            "Sorry for this response, we will improve the system!")

    # Chat input at the bottom
    st.markdown("---")
    col_input, col_button = st.columns([4, 1])
    with col_input:
        question = st.text_input("Type your message...", key="chat_input")
    with col_button:
        if st.button("➡️ Send"):
            if question:
                response = requests.post(API_URL, json={"question": question}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    chatbot_response = data.get(
                        'answer', "Answer not available.")
                    sources = data.get('source', "Unknown source")
                    focus_area = data.get('focus_area', "Not specified")
                    similarity_score = data.get('similarity', "N/A")
                    similarity_type = data.get('similarity_type', "N/A")

                    st.session_state.history.append(
                        {
                            "question": question,
                            "response": chatbot_response,
                            "sources": sources,
                            "focus_area": focus_area,
                            "similarity": similarity_score,
                            "similarity_type": similarity_type,
                            "metrics": data.get("metrics", {}),
                            "feedback": None,
                        }
                    )
                else:
                    st.error("Error retrieving the answer.")

# ------------------- 📊 FEEDBACK & ANALYSIS -------------------
with tabs[1]:
    generate_and_display_graphs()
