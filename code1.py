import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq  # Updated import for Groq
import traceback

# You'll need to create this file or inline the templates
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):

    try:
        # Use a robust embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain(vectorstore):

    try:
        # Use Groq LLM with configurable parameters
        llm = ChatGroq(
            temperature=0.5,
            model_name="llama3-70b-8192",  # You can change the model as needed
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def handle_userinput(user_question):
    """
    Handle user input and display chat history.
    
    Args:
        user_question (str): User's input question
    """
    try:
        # Check if conversation is initialized
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first.")
            return

        # Get the chatbot response
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Check if the response indicates no relevant information
        if not response['chat_history'] or len(response['chat_history']) < 2:
            # Fallback response
            fallback_message = (
                "Sorry, I couldn't find relevant information in the documents. "
                "Would you like to rephrase your question, or connect with a live agent?"
            )
            st.write(bot_template.replace(
                "{{MSG}}", fallback_message
            ), unsafe_allow_html=True)
            return

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        # Log error and provide a fallback message
        st.error(f"Error processing user input: {e}")
        traceback.print_exc()

        fallback_message = (
            "Something went wrong while processing your request. "
            "Would you like to try again or connect with a live agent?"
        )
        st.write(bot_template.replace(
            "{{MSG}}", fallback_message
        ), unsafe_allow_html=True)

def main():

    load_dotenv()
    st.set_page_config(page_title="AI Chatbot Assignment")
    st.write(css, unsafe_allow_html=True)

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    st.header("Chat with Your PDFs 📚")
    
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Document Library")
        pdf_docs = st.file_uploader(
            "Upload PDFs here", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        

        if st.button("Process Documents"):
            with st.spinner("Processing Documents..."):
                try:
                    
                    if not pdf_docs:
                        st.warning("Please upload at least one PDF.")
                        return

                    
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text:
                        st.warning("No text could be extracted from the PDFs.")
                        return

                   
                    text_chunks = get_text_chunks(raw_text)
                    
                    
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore is None:
                        st.error("Failed to create vector store.")
                        return

                    
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success(f"Successfully processed {len(pdf_docs)} PDF(s)!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    traceback.print_exc()

if __name__ == '__main__':
    main()