from flask import Flask, render_template, request
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import RetrievalQA
import PyPDF2
import json
import os

app = Flask(__name__)


f = open('openaikey.txt')
os.environ["OPENAI_API_KEY"]=f.read()
# Initialize the LLM 
llm = OpenAI(temperature=0,max_tokens=1500)

# Initialize embeddings using OpenAI
embeddings = OpenAIEmbeddings()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload and display content
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.pdf'):
        # Read the PDF content
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        compliance_verdicts = check_compliance(text, standard_vectorstore)
        print(compliance_verdicts)

        # Render the content on the webpage
        return render_template('display.html', content=compliance_verdicts['response'], costout=compliance_verdicts['costout'])
    else:
        return "Please upload a valid PDF file"

def load_and_embed_document(pdf_path):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Embed the documents and create a vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore

# Example: Load and embed the standard document
standard_document_path = "resources/DesignDocumentStandards.pdf"
standard_vectorstore = load_and_embed_document(standard_document_path)

def check_compliance(content, vectorstore):
    # Load the user document using PyPDFLoader
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    
    # Generate compliance checks for each sentence in the user document
    compliance_checks = []
    
    query=f"""
    Given the knowledge base you have help me reivew the content mentioned enclosed between <start> and <end> tag and provide your review comments
    <start>
    {content}
    <end>
    Please also highlight grammatically incorrect statements and abmiguous statements.
    Validate if all the necessary sections are present, specify which sections are missing
    Finally review the content and provide the feedback checking if the document is compliant with the standards defined.
    
    Also give the final verdict as "Approved", only if it met all the compliances or "Rejected" in the format "Final Verdit : <Verdict>
    """
    with get_openai_callback() as cb:
        response = qa_chain.invoke(query)
        responseout = {}
        responseout["response"] = response['result']
        responseout["costout"] = cb
        print(responseout)
        return responseout




if __name__ == '__main__':
    app.run(debug=True)
