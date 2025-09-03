import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from utils import ensure_event_loop
import google.generativeai as genai
import time
import uuid

class PDFChatbot:
    def __init__(self, google_api_key):
        self.google_api_key = google_api_key
        self.vectorstore = None
        self.retriever = None

    def extract_text(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {e}")

    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        """Split text into smaller chunks for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def embed_and_store(self, texts, max_retries=3):
        """Embed text chunks and store them in a FAISS vector store."""
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
        for attempt in range(max_retries):
            try:
                self.vectorstore = FAISS.from_texts(texts, embeddings)
                self.retriever = self.vectorstore.as_retriever()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Embedding or FAISS storage failed after retries: {e}")
    
    def save_vectorstore(self, pdf_file_name, cache_dir="faiss_cache"):
        """Save the FAISS vector store to a local directory."""
        if not self.vectorstore:
            return
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Use a unique name for the cache based on the PDF filename
        cache_file_path = os.path.join(cache_dir, f"{pdf_file_name}.faiss")
        self.vectorstore.save_local(cache_file_path)
        print(f"Vector store saved to {cache_file_path}")

    def load_vectorstore(self, pdf_file_name, cache_dir="faiss_cache"):
        """Load the FAISS vector store from a local directory."""
        cache_file_path = os.path.join(cache_dir, f"{pdf_file_name}.faiss")
        if os.path.exists(cache_file_path):
            print(f"Loading vector store from cache: {cache_file_path}")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            self.vectorstore = FAISS.load_local(
                cache_file_path,
                embeddings,
                allow_dangerous_deserialization=True # This is required for loading
            )
            self.retriever = self.vectorstore.as_retriever()
            return True
        return False

    def ask(self, query):
        """Query the PDF content using Gemini API directly with retrieved context."""
        if not self.retriever:
            raise RuntimeError("No retriever found. Please upload and process a PDF first.")
        try:
            ensure_event_loop()
            # Retrieve relevant context from the vector store
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Prepare the prompt for Gemini
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            # Set up Gemini client
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Call Gemini API
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to get answer: {e}")