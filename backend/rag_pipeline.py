import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# ======== Global Setup ======== #

def build_pipeline():
    # Load and flatten Excel
    df = pd.read_excel("network_elements_with_yearwise_versions.xlsx", sheet_name="Network Elements", header=[0, 1]).fillna("Unknown")
    df.columns = [f"{sec.strip()} - {param.strip()}" for sec, param in df.columns]

    # Convert each row to a document
    docs = []
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append({"content": text})
    df_docs = pd.DataFrame(docs)

    loader = DataFrameLoader(df_docs, page_content_column="content")
    documents = loader.load()

    # Split for better retrieval
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    documents = splitter.split_documents(documents)

    # BM25
    bm25 = BM25Retriever.from_documents(documents, k=5)

    # Embeddings + Chroma
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding=embedder, persist_directory="chroma_store")
    vect_retr = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Combine both retrievers
    hybrid = EnsembleRetriever(retrievers=[bm25, vect_retr], weights=[0.5, 0.5])

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyAhafspBPpv48ReYM7qzEdGVE7Tzt-FiF4")

    # Custom prompt
    prompt = PromptTemplate.from_template("""
    Answer the question based on the following network configuration data. If not found, say "Not found".
    <context>
    {context}
    </context>
    Question: {question}
    """)

    document_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        document_variable_name="context"
    )

    qa_chain = RetrievalQA(
        retriever=hybrid,
        combine_documents_chain=document_chain,
        return_source_documents=False  # or True if you want to debug
    )

    return qa_chain

# Initialize only once
qa_chain = build_pipeline()

# ======== Interface function for Flask ======== #
def answer_question(query: str) -> str:
    try:
        result = qa_chain.invoke({"query": query})
        if isinstance(result, dict):
            return result.get("answer") or result.get("result") or "No answer found."
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
