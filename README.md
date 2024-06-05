## 🦙💡 Local AI Workbench

This repository houses a versatile Streamlit application designed for interacting with a large language model (LLM) running locally. You can choose to use **Ollama** or **LM Studio** to manage and run your LLM.

**Features:**

- **Chat Interface:** Engage in natural language conversations with your local LLM.
- **Data Analysis:** Utilize PandasAI for powerful data analysis with natural language prompts. Generate insights and visualizations from your data.
- **Code Generation:** Generate code in various programming languages based on your descriptions. 
- **Image Analysis:** Perform basic image processing tasks like grayscale conversion, edge detection, blurring, contour finding, and histogram analysis directly within the app.
- **Text Analysis:** Analyze text for sentiment, extract words and sentences.

## 🚀 Getting Started

### Prerequisites

1. **Python 3.7+:** Ensure you have Python installed.
2. **Virtual Environment (Recommended):** Create a virtual environment to manage dependencies.
   ```bash
   python -m venv env
   source env/bin/activate 
   ```
3. **Ollama OR LM Studio:**
   - **Ollama:** Download and install from [https://ollama.ai/](https://ollama.ai/) if you prefer a command-line based approach.
   - **LM Studio:** Download and install from [https://lmstudio.com/](https://lmstudio.com/) for a more visual and user-friendly interface.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/local-ai-workbench.git
   cd local-ai-workbench
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Download a Model:**  
   - **Using Ollama:** Use the Ollama CLI to download a model, e.g., `ollama pull quantfactory/meta-llama-3-8b-instruct-gguf`.
   - **Using LM Studio:** Download the "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" model (or any compatible model) directly within the LM Studio application. 
2. **Start the Model:** 
   - **Ollama:** Launch the downloaded model using the Ollama CLI, e.g., `ollama run quantfactory/meta-llama-3-8b-instruct-gguf`.
   - **LM Studio:** Start the model within the LM Studio interface. 
3. **Update `app.py`:**
   - **API_URL:** 
      - **Ollama:** Set this to your Ollama server address, typically `http://localhost:1234/v1`.
      - **LM Studio:** Set this to the API endpoint provided by LM Studio for your running model.
   - **MODEL_NAME:** Ensure this matches the model name exactly.

### Running the App

```bash
streamlit run My_Chatbot.py
```

Open your web browser and navigate to the address displayed in the terminal, usually `http://localhost:8501`.

## 🙏 Acknowledgments

Special thanks to:

- **Tirendaz AI:** The inspiration for this project came from the insightful YouTube video: [How to Use Llama 3 with PandasAI and Ollama Locally](https://youtu.be/_dDaNgBDoHY?si=QeSwphhbrVtQE_cU). 


## Requirements 

```
streamlit
pandas
opencv-python
Pillow
matplotlib
pandasai
openai
textblob
```

