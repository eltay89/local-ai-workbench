import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
from openai import OpenAI
from textblob import TextBlob

# --- App Configuration ---
API_URL = "http://localhost:1234/v1"
MODEL_NAME = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"

def setup_app():
    """Configures the Streamlit app with a title, icon, layout, and custom styles."""
    st.set_page_config(
        page_title="AI-Powered Local Language Model Assistant",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
        """
        <style>
        body {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        .reportview-container .main .block-container {
            max-width: 95%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .reportview-container .main {
            background-color: #f4f4f4;
            color: #212529; 
            border-radius: 10px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        }
        .stButton button, .stDownloadButton button, .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {
            background-color: #1f77b4;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
        }
        .stTextInput, .stTextArea {
            width: 90%;
            padding: 10px;
        }
        .stSelectbox, .stFileUploader {
            width: 100%;
        }
        .stMarkdown {
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #343a40;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return get_llm_connection()

@st.cache_resource
def get_llm_connection():
    """Establishes and caches the connection to the local LLM."""
    client = OpenAI(base_url=API_URL, api_key="lm-studio")
    llm = LocalLLM(api_base=API_URL, model=MODEL_NAME)
    return client, llm

def handle_chat(client, llm):
    """Handles the chat interaction between the user and the local LLM."""
    st.header("Chat with Your Local LLM")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.text_input("Your message:", key="chat_input")

    if st.button("Send"):
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=llm.model,
                        messages=st.session_state.messages,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error communicating with the model: {e}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

def handle_data_analysis(client, llm):
    """Handles the data analysis functionality using PandasAI."""
    st.header("Data Analysis with PandasAI")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, sheet_name=None)

            if isinstance(data, dict):
                selected_sheet = st.sidebar.selectbox("Select a sheet:", list(data.keys()))
                data = data[selected_sheet]
                st.write(f"First 5 rows of '{selected_sheet}':")
            else:
                st.write("First 5 rows:")
            st.write(data.head().style.set_properties(**{'background-color': '#f0f0f0', 'color': '#212529'}))

            df = SmartDataframe(data, config={"llm": llm})

            prompt_history = st.container()
            
            with prompt_history:
                prompt = st.text_area("Enter Your Prompt:")
                if st.button("Analyze"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Ensure that the prompt is a string
                            prompt = str(prompt)
                            response = df.chat(prompt)
                            # Ensure that the response is a string before calling startswith
                            if isinstance(response, str) and response.startswith("## Plot"):
                                df.plot(prompt)
                            else:
                                st.write(response)
                        except Exception as e:
                            st.error(f"Analysis error: {e}")

        except Exception as e:
            st.error(f"Error loading file: {e}")

def handle_code_generation(client, llm):
    """Handles the code generation functionality using the LLM."""
    st.header("Code Generation")
    prompt = st.text_area("Describe the code you want to generate:")
    if st.button("Generate Code"):
        with st.spinner("Generating code..."):
            try:
                response = client.chat.completions.create(
                    model=llm.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                st.code(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Code generation error: {e}")

def handle_image_analysis(client, llm):
    """Handles local image analysis, offering various image processing options."""
    st.header("Local Image Analysis")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        opencv_image = np.array(image.convert('RGB'))
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        option = st.selectbox("Choose an image processing action:", [
            "Select", "Grayscale", "Edges", "Blur", "Contours", "Histogram"
        ])

        if option == "Grayscale":
            grayscale = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            st.image(grayscale, caption='Grayscale Image', use_column_width=True, channels="L")

        elif option == "Edges":
            edges = cv2.Canny(opencv_image, 100, 200)
            st.image(edges, caption='Edge Detection', use_column_width=True, channels="L")

        elif option == "Blur":
            blur = cv2.GaussianBlur(opencv_image, (11, 11), cv2.BORDER_DEFAULT)
            st.image(blur, caption='Blurred Image', use_column_width=True)

        elif option == "Contours":
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = cv2.drawContours(opencv_image.copy(), contours, -1, (0, 255, 0), 2)
            st.image(contour_image, caption='Contours Detected', use_column_width=True)

        elif option == "Histogram":
            fig, ax = plt.subplots()
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([opencv_image], [i], None, [256], [0, 256])
                ax.plot(histr, color=col)
                ax.set_xlim([0, 256])
            st.pyplot(fig)

def handle_text_analysis(client, llm):
    """Handles the text analysis functionality using TextBlob."""
    st.header("Text Analysis")
    text = st.text_area("Enter text for analysis:")
    if st.button("Analyze Text"):
        with st.spinner("Analyzing text..."):
            try:
                blob = TextBlob(text)
                st.write("Sentiment: ", blob.sentiment)
                st.write("Words: ", blob.words)
                st.write("Sentences: ", blob.sentences)
            except Exception as e:
                st.error(f"Text analysis error: {e}")

def main():
    """The main function to run the Streamlit application."""
    client, llm = setup_app()
    modes = {
        "Chat": handle_chat,
        "Data Analysis": handle_data_analysis,
        "Code Generation": handle_code_generation,
        "Image Analysis": handle_image_analysis,
        "Text Analysis": handle_text_analysis  # New mode
    }
    
    mode = st.sidebar.selectbox("Select Mode:", list(modes.keys()))
    modes[mode](client, llm)

    # Developer Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by: Mohamed")
    st.sidebar.markdown("GitHub: eltay89")

if __name__ == "__main__":
    main()
