# app.py
from flask import Flask, render_template, request, jsonify
import uuid
import base64
import os
from dotenv import load_dotenv
import threading

import time
# Import necessary libraries for RAG pipeline
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables to store our pipeline components
vectorstore = None
store = None
retriever = None
chain = None
chain_with_sources = None

# Initialize the RAG pipeline
def initialize_pipeline():
    global vectorstore, store, retriever, chain, chain_with_sources
    
    print("Initializing embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Setting up vector store...")
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_function)
    
    # Storage layer for parent documents
    store = InMemoryStore()
    id_key = 'doc_id'
    
    # Create the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    print("Setting up LLM...")
    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    
    # Set up the LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )
    
    # Define helper functions
    def parse_docs(docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                base64.b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}
    
    def build_prompt(kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs['question']
    
        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text + " "
    
        # Truncate context to avoid exceeding model limits
        max_context_length = 1000
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."
    
        prompt_template = f"""
    Answer the question in a humble way based only on the following context, which can include text, tables, and the below images. It should not contain irrelevant answers.
    Context: {context_text}
    Question: {user_question}
    """
    
        return prompt_template.strip()
    
    # Set up the chains
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
    
    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | llm
            | StrOutputParser()
        )
    )
    
    print("RAG pipeline initialized.")

# Process uploaded PDF file
def process_pdf(file_path):
    print(f"Processing PDF: {file_path}")
    
    # Partition the PDF
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        chunking_strategy='by_title',
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    # Separate text chunks
    texts = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    # Get tables
    tables = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Table" in str(type(el)):
                    tables.append(el)
    
    # Get images
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    
    print(f"Extracted {len(texts)} text chunks, {len(tables)} tables, and {len(images)} images")
    
    # Create summaries
    # For text and tables, use Groq
    prompt_text = """
    You are an assistant tasked with summarizing text and tables.
    Give a concise summary of the table or text.
    
    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is your summary" or anything like that.
    
    Table or text chunk : {element}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.3-70b-versatile")
    summarize_chain = prompt | model | StrOutputParser()
    
    print("Generating text summaries...")
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    
    time.sleep(2) 
    
    
    print("Generating table summaries...")
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    time.sleep(4)
    
    # For images, use Gemini
    print("Generating image summaries...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    mime_type = "image/jpeg"
    text_query = """
    Describe the image in detail. For context,
    the image is a part of research paper explaining the content.
    Be specific about graphs, such as bar plots
    """
    
    image_summaries = []
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    for base64_image_data in images:
        contents = [
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_image_data
                }
            },
            {
                "text": text_query
            }
        ]
        
        try:
            response = model.generate_content(contents)
            image_summaries.append(response.text)
        except Exception as e:
            print(f"An error occurred summarizing image: {e}")
            image_summaries.append("Error generating image summary")
    
    # Add to vector store
    print("Adding to vector store...")
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={retriever.id_key: doc_ids[i]}) 
        for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    
    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={retriever.id_key: table_ids[i]}) 
        for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={retriever.id_key: img_ids[i]}) 
        for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
    
    print("PDF processing complete")
    return len(texts), len(tables), len(images)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the file
        temp_file_path = f"uploads/{str(uuid.uuid4())}.pdf"
        os.makedirs("uploads", exist_ok=True)
        file.save(temp_file_path)
        
        # Process in a separate thread to not block the response
        def process_thread():
            try:
                text_chunks, tables, images = process_pdf(temp_file_path)
                print(f"Processed {text_chunks} text chunks, {tables} tables, and {images} images")
            except Exception as e:
                print(f"Error processing PDF: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        threading.Thread(target=process_thread).start()
        
        return jsonify({'message': 'File uploaded and processing started'}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get response with context
        response_with_context = chain_with_sources.invoke(question)
        print(response_with_context)
        
        # Format result
        result = {
            'answer': response_with_context['response'],
            'sources': {
                'images': [],
                'texts': []
            }
        }
        
        # Process text sources
        for text_source in response_with_context['context']['texts']:
            result['sources']['texts'].append({
                'content': text_source.text,
                'page_number': text_source.metadata.page_number if hasattr(text_source.metadata, 'page_number') else None
            })
        
        # Process image sources
        for image_source in response_with_context['context']['images']:
            result['sources']['images'].append(image_source)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the pipeline
    initialize_pipeline()
    app.run(debug=True)