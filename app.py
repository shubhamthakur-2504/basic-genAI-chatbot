from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpfull assistant. please response to the users question."),
        ("user", "question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature):

    llm = ChatGoogleGenerativeAI(model=engine, api_key=api_key, temperature=temperature)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke({"question": question})

    return response


# --- Streamlit App ---

st.set_page_config(page_title="Q&A Chatbot", layout="wide")
st.title("🤖 Gemini Q&A Chatbot")

# Sidebar for API Key and Model Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Enter your Gemini API Key", type="password", key="api_key_input")
    
    # Model selection with user-friendly names
    selected_model = st.selectbox(
        "Select LLM Model",
        [
            "gemini-2.5-flash", # Fast and cost-effective
            "gemini-2.5-pro",  # Advanced multimodal model
        ],
        key="model_selector"
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key="temp_slider")

# Main app interface
user_question = st.text_input("Ask me anything:", key="user_question_input")

if st.button("Get Answer", key="get_answer_button"):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner(f"Asking {selected_model}..."):
            try:
                response = generate_response(user_question, api_key, selected_model, temperature)
                st.success("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")