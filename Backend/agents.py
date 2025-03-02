"""
This module contains utility functions
for evaluating and logging metrics for the Gemini AI model,
including OCR-based text extraction from medication images.
"""

from typing import List
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import API_KEY

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, google_api_key=API_KEY)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")


def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from a medication image using TrOCR.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Extracted text from the image.
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return extracted_text


def generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for the given text."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def generate_response(question: str, context: str, language: str) -> str:
    """Generate an enriched response using Gemini AI model."""
    prompt_template = ChatPromptTemplate.from_template("""
    You are a medical AI assistant with expertise in clinical studies.
    Your goal is to provide accurate and structured answers.

    **Instructions :**
    - Prioritize medically validated information.
    - If the context is unclear, clarify before answering.
    - Use clear, professional language.
    - Cite sources if available.

    **Question:** {question}
    **Context:** {context}
    **Language:** {language}
    """)

    chain = prompt_template | llm
    response = chain.invoke({
        "question": question,
        "context": context,
        "language": language
    })
    return response.content


def correct_medication_name(medication_text: str) -> str:
    """
    Uses Gemini AI to correct OCR errors in the extracted medication name.
    """
    prompt_template = ChatPromptTemplate.from_template("""
    You are an AI specialized in medication data correction.
    Your task is to correct the name of a medication that may have errors due to OCR mistakes.

    **Instructions:**
    - If the extracted name is misspelled, correct it.
    - If it is ambiguous, return the closest known medication.
    -Prioritize well-khnow pharmaceutical brands and medecine
    - Do not add extra words or explanations, return only the corrected name.

    **Extracted Medication Name:** {medication_text}
    """)

    chain = prompt_template | llm
    response = chain.invoke({"medication_text": medication_text})
    
    return response.content.strip()


def get_medication_details(medication_name: str, language: str) -> str:
    """Uses Gemini AI to provide practical information about a medication."""
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are a medical AI assistant with expertise in pharmaceuticals.
    Provide **practical** and **concise** information about the given medication.

    **Instructions:**
    - Clearly explain why this medication is prescribed.
    - List contraindications (who should not take it).
    - Mention common and serious side effects.
    - If applicable, suggest precautions or interactions with other drugs.
    - Ensure accuracy and use reliable medical knowledge.

    **Medication:** {medication_name}
    **Language:** {language}
    """)

    chain = prompt_template | llm
    response = chain.invoke({"medication_name": medication_name, "language": language})
    
    return response.content


