"""
This module serves as the main Streamlit application for the Medical Chatbot.
It provides an intuitive user interface for interacting with the chatbot, analyzing medical images, and collecting feedback for further improvements.
"""

import streamlit as st
import requests
from graph import generate_and_display_graphs
from agents import generate_response, get_medication_details
from config import API_KEY

# ------------------- CONFIGURATION -------------------
API_URL = "http://127.0.0.1:8000/answer"

# ------------------- STREAMLIT INTERFACE -------------------
st.set_page_config(page_title="Patient Assistant", page_icon="🩺", layout="wide")

# Global Styling to ensure uniformity across all tabs
st.markdown(
    """
    <style>
        /* Centering the title, headings, and paragraphs */
        h1, h2, p, .stMarkdown {
            text-align: center !important;
        }
        
        /* Center the tabs section */
        div[data-testid="stTabs"] {
            display: flex;
            justify-content: center;
        }

        /* Center the content inside the tabs */
        .tab-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            width: 100%;
            margin: 0 auto;
        }

        /* Centering text input, file uploaders, and buttons */
        .stTextInput, .stFileUploader, .stButton {
            margin: 10px auto;
            display: block;
            width: 80%;
        }

        /* Ensuring images within the tabs are also centered */
        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* Styling the tabs for a more centered alignment */
        .stTabs {
            display: flex;
            justify-content: center;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("<h1>🩺 Patient Assistant</h1>", unsafe_allow_html=True)

# Tabs Section
tabs = st.tabs([
    "🏠 About the Project",
    "💬 Chatbot",
    "🖼️ Medical Image Analysis",
    "📊 Feedback & Analysis"
])

# ------------------- 🏠 ABOUT THE PROJECT -------------------
with tabs[0]:
    st.markdown(
        """
        <div class='tab-content'>
            <h2>📌 About the Project</h2>
            <p>
                <b>Patient Assistant</b> is designed to help users ask medical questions.
                Get insights on symptoms, analyze radiological images, and obtain information about medications.
                <br><br>
                <i>While this AI-powered assistant provides useful advice, always consult a healthcare professional for final decisions.</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------- 💬 CHATBOT -------------------
with tabs[1]:
    st.markdown(
        """
        <div class='tab-content'>
            <h2>💬 Chat with the Medical AI</h2>
            <p>Ask a question and receive an instant AI-powered response.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "history" not in st.session_state:
        st.session_state.history = []
   
    for chat in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(f"**🗨️ {chat['question']}**")
        with st.chat_message("assistant"):
            st.markdown(f"**📝 Answer:** {chat['response']}")
            st.markdown(f"🔍 **Source:** {chat['sources']}")
            st.markdown(f"📌 **Focus Area:** {chat['focus_area']}")
            st.markdown(f"💡 **Similarity Score:** {chat['similarity']} ({chat['similarity_type']})")
   
    question = st.chat_input("Type your message...")
    if question:
        response = requests.post(API_URL, json={"question": question}, timeout=500)
        if response.status_code == 200:
            data = response.json()
            chatbot_response = data.get('answer')
            st.session_state.history.append(
                {
                    "question": question,
                    "response": chatbot_response,
                    "sources": data.get('source', "Unknown source"),
                    "focus_area": data.get('focus_area', "Not specified"),
                    "similarity": data.get('similarity', "N/A"),
                    "similarity_type": data.get('similarity_type', "N/A"),
                }
            )
        else:
            chatbot_response = generate_response(question, None, "english")
            st.session_state.history.append(
                {
                    "question": question,
                    "response": chatbot_response,
                    "sources": "Unknown source",
                    "focus_area": "Not specified",
                    "similarity": "N/A",
                    "similarity_type": "N/A",
                }
            )

# ------------------- 🖼️ MEDICAL IMAGE ANALYSIS -------------------
with tabs[2]:
    st.markdown(
        """
        <div class='tab-content'>
            <h2>🩺 Medical Image Analysis</h2>
            <p>Upload an image to identify medications.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image")

        with st.spinner("Processing Medication Image..."):
            files = {"image": uploaded_image.getvalue()}
            response = requests.post("http://127.0.0.1:8000/process_medication_image", files=files)

            if response.status_code == 200:
                data = response.json()
                corrected_name = data.get("corrected_name", "Unknown")
                medication_info = data.get("medication_info", "No information available.")

                st.subheader("🔍 Analysis Result")
                st.write(f"**Corrected Medication Name:** {corrected_name}")

                # Demander à l'utilisateur s'il valide le nom
                user_confirmation = st.radio("Is this medication name correct?", ("Yes", "No"), index=0)

                if user_confirmation == "Yes":
                    st.write(medication_info)  # Affiche directement les infos
                else:
                    # L'utilisateur peut entrer le nom correct
                    user_corrected_name = st.text_input("Enter the correct medication name")

                    if user_corrected_name:
                        if st.button("Generate Updated Medication Info"):
                            response_2 = get_medication_details(user_corrected_name,"english")
                            st.write(response_2)
                            
            else:
                st.error("Error processing the image.")






# ------------------- 📊 FEEDBACK & ANALYSIS -------------------
with tabs[3]:
    st.markdown(
        """
        <div class='tab-content'>
            <h2>📊 Feedback & Analysis</h2>
            <p>Review collected feedback and data-driven insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    generate_and_display_graphs()
