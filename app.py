import streamlit as st
from PyPDF2 import PdfReader
import os
import requests
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

# Load environment variables
GEMINI_API_KEY = "AIzaSyBbhY5dG1OgIzHwS5sK4TVvxS7pjFYyRQI"
genai.configure(api_key=GEMINI_API_KEY)

# Web scraping headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'
}

def get_gemini_response(stock_name):
    """Get and store Gemini API response for the stock"""
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        # Prompt for stock analysis
        prompt = f"Provide a detailed analysis of {stock_name} including key metrics, performance, and market position. Format as JSON."
        
        # Get response
        response = model.generate_content(prompt)
        
        # Save response
        os.makedirs("APIData", exist_ok=True)
        filename = f"APIData/{stock_name}_gemini_analysis.json"
        
        with open(filename, 'w') as f:
            json.dump({"analysis": response.text}, f, indent=4)
            
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini analysis: {e}")
        return ""

def merge_data(pdf_text, api_text):
    """Merge PDF and API data into a single context"""
    merged_text = f"""
API Analysis:
{api_text}

PDF Content:
{pdf_text}
"""
    # Save merged content
    os.makedirs("MergedData", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = f"MergedData/merged_context_{timestamp}.txt"
    
    with open(merged_file, 'w', encoding='utf-8') as f:
        f.write(merged_text)
        
    return merged_text

class DocumentDownloader:
    def __init__(self, stock_name):
        self.stock_name = stock_name
        self.base_url = f"https://www.screener.in/company/{stock_name}/consolidated/"
        self.ppt_folder = "PPTFiles"
        os.makedirs(self.ppt_folder, exist_ok=True)

    def fetch_page(self):
        try:
            response = requests.get(self.base_url, headers=HEADERS)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching page: {e}")
            return None

    def get_ppt_links(self):
        html_content = self.fetch_page()
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, "html.parser")
        ppt_links = []
        
        for link in soup.find_all("a", href=True, class_="concall-link"):
            href = link["href"]
            text = link.get_text(strip=True).lower()
            if "ppt" in text:
                if not href.startswith("http"):
                    href = f"https://www.screener.in{href}"
                ppt_links.append(href)
        
        return ppt_links[:1]  # Return only the latest PPT link

    def download_ppts(self):
        ppt_links = self.get_ppt_links()
        downloaded_files = []
        
        for i, url in enumerate(ppt_links):
            filename = f"{self.stock_name}_ppt_{i+1}.pdf"
            filepath = os.path.join(self.ppt_folder, filename)
            
            if os.path.exists(filepath):
                st.info(f"Already downloaded: {filename}")
                downloaded_files.append(filepath)
                continue
            
            try:
                response = requests.get(url, headers=HEADERS)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filepath)
                st.success(f"Downloaded: {filename}")
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
        
        return downloaded_files

# RAG Components
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant answering questions based on provided documents.
    Use the conversation history for better context.
    If the answer is not available, respond with "answer is not available in the context."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GEMINI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_query(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Stock Data Processor with RAG", layout="wide")
    st.header("Retrieve and Process Stock Data")

    with st.sidebar:
        stock_name = st.text_input("Enter Stock Symbol (e.g., TATAMOTORS)", "TATAMOTORS")
        if st.button("Download and Process Data"):
            # 1. First get API data
            with st.spinner("Fetching API analysis..."):
                api_text = get_gemini_response(stock_name)
            
            # 2. Then download PDFs
            downloader = DocumentDownloader(stock_name)
            pdf_paths = downloader.download_ppts()
            
            if pdf_paths:
                with st.spinner("Processing data..."):
                    # 3. Get PDF text
                    pdf_text = get_pdf_text(pdf_paths)
                    
                    # 4. Merge data
                    merged_text = merge_data(pdf_text, api_text)
                    
                    # 5. Process for RAG
                    text_chunks = get_text_chunks(merged_text)
                    get_vector_store(text_chunks)
                    st.success("All data processed and stored in vector database!")
            else:
                st.warning("No PDFs found for processing.")
        
    user_question = st.text_input("Ask a question about the stock reports")
    if user_question:
        if os.path.exists("faiss_index"):
            response = user_query(user_question)
            st.write(response)
        else:
            st.warning("Please process data first before asking questions.")

if __name__ == "__main__":
    main()