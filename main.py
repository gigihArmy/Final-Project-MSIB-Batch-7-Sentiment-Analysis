import libraries as lib  # libraries.py
import eda  # file khusus eksplorasi dataset
import classification  # file untuk klasifikasi sentimen
import gemini_flash_llm  # file untuk llm

# Membuat sidebar
lib.st.sidebar.title("Final Project Sentiment Classification")
page = lib.st.sidebar.radio("Navigation:", ["Exploration Data Analysis", "Sentiment Classification", "Gemini Flash LLM"])

# Logika untuk memuat file sesuai pilihan
if page == "Exploration Data Analysis":
    eda.run()
elif page == "Sentiment Classification":
    classification.run()
elif page == "Gemini Flash LLM":
    gemini_flash_llm.run()
