import libraries as lib

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def run():
    
    def chat(contexts, history, question):
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=API_KEY
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert assistant in text analysis. Your task is to analyze text data and extract the main topics or themes discussed from given data.",
                ),
                (
                    "human",
                    """
                    This is the data: {contexts}

                    Your task:
                    1. Extract the main topics by identifying recurring themes, keywords, or phrases.
                    2. Provide the following:
                       - Elaborate of the main topics.
                       - Three common keywords or phrases.

                    Instructions:
                        - If Positive: Focus on benefits, advantages, praises, or successful experiences.
                        - If Negative: Focus on complaints, criticisms, issues, or failures.
                        - If Neutral: Focus on factual information, objective observations, or undecided opinions.

                    Use this chat history to generate relevant answers from recent conversation: {history}

                    User question: {question}
                    """,
                ),
            ]
        )
    
        chain = prompt | llm
        completion = chain.invoke(
            {
                "contexts": contexts,
                "history": history,
                "question": question,
            }
        )

        answer = completion.content
        input_tokens = completion.usage_metadata['input_tokens']
        completion_tokens = completion.usage_metadata['output_tokens']

        result = {}
        result["answer"] = answer
        result["input_tokens"] = input_tokens
        result["completion_tokens"] = completion_tokens
        return result
    
    lib.st.title("Gemini Flash Chatbot")
    lib.st.write("LLM ini digunakan untuk mengekstrak topik dari hasil klasifikasi sentimen.")
    
    API_KEY = lib.st.text_input("Masukkan API Key:", type="password")

    if lib.st.button("Submit"):
        # Fungsi untuk mengecek validitas API Key
        def check_api_key(api_key):
            try:
                # Cek validitas dengan mencoba inisialisasi ChatGoogleGenerativeAI
                ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.7,
                    api_key=api_key
                )
                lib.st.success("API Key valid!")
                return True
            except Exception as e:
                lib.st.error(f"API Key tidak valid: {str(e)}")
                return False
        
    if API_KEY is not None:
        uploaded_file = lib.st.file_uploader('Upload Dataset CSV', type='csv')
    
        if uploaded_file is not None:
            lib.st.success('File telah terupload')
            df = lib.pd.read_csv(uploaded_file)
            contexts = df.to_string()
            lib.st.write('Dataframe')
            lib.st.write(df.head())
            lib.st.write(df.shape)

            # Initialize chat history
            if "messages" not in lib.st.session_state:
                lib.st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in lib.st.session_state.messages:
                with lib.st.chat_message(message["role"]):
                    lib.st.markdown(message["content"])

            # React to user input
            if prompt := lib.st.chat_input("What is up?"):
                # Get chat history if not Null
                messages_history = lib.st.session_state.get("messages", [])[-10:]
                history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

                # Display user message in chat message container
                lib.st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                lib.st.session_state.messages.append({"role": "user", "content": prompt})

                response = chat(contexts, history, prompt)
                answer = response["answer"]
                input_tokens = response["input_tokens"]
                completion_tokens = response["completion_tokens"]

                # Display assistant response in chat message container
                with lib.st.chat_message("assistant"):
                    lib.st.markdown(answer)
                    container = lib.st.container(border=True)
                    container.write(f"Input Tokens : {input_tokens}")
                    container.write(f"Completion Tokens: {completion_tokens}")
    
                # Display history chat
                with lib.st.expander("See Chat History"):
                    lib.st.write("**History Chat:**")
                    lib.st.code(history)

                # Add assistant response to chat history
                lib.st.session_state.messages.append({"role": "assistant", "content": answer})
            
        else:
            lib.st.warning('Upload dataset terlebih dahulu!')
            
    else:
        lib.st.warning('Masukkan API Key terlebih dahulu')