import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# ======== Prompt Templates ======== #
prompt_templates = {
    "summary": ChatPromptTemplate.from_template(
        """Summarize the following content in bullet points:\n<context>\n{context}\n</context>"""
    ),
    "explanation": ChatPromptTemplate.from_template(
        """Explain in detail using only the provided context:\n<context>\n{context}\n</context>\nQuestion: {input}"""
    ),
    "factual": ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context.\n<context>\n{context}\n</context>\nQuestion: {input}"""
    ),
    "json": ChatPromptTemplate.from_template(
        """Based only on the following context, generate a structured JSON answer for the given NE:\n<context>\n{context}\n</context>\nQuestion: {input}"""
    )
}

def build_pipeline(prompt_mode="factual"):
    df = pd.read_excel("network_elements_with_yearwise_versions.xlsx", sheet_name="Network Elements", header=[0, 1]).fillna("Unknown")
    df.columns = [f"{sec.strip()} - {param.strip()}" for sec, param in df.columns]

    docs = []
    for _, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append({"content": text})
    df_docs = pd.DataFrame(docs)

    loader = DataFrameLoader(df_docs, page_content_column="content")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    documents = splitter.split_documents(documents)

    bm25 = BM25Retriever.from_documents(documents, k=5)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding=embedder, persist_directory="chroma_store")
    vect_retr = vectorstore.as_retriever(search_kwargs={"k": 5})

    hybrid = EnsembleRetriever(retrievers=[bm25, vect_retr], weights=[0.5, 0.5])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="GOOGLE_API_KEY")

    selected_prompt = prompt_templates.get(prompt_mode, prompt_templates["factual"])

    document_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=selected_prompt),
        document_variable_name="context"
    )

    qa_chain = RetrievalQA(
        retriever=hybrid,
        combine_documents_chain=document_chain,
        return_source_documents=False
    )

    return qa_chain


qa_chain_cache = {
    "factual": build_pipeline("factual")
}

def answer_question(query: str, mode="factual") -> str:
    try:
        if mode not in qa_chain_cache:
            qa_chain_cache[mode] = build_pipeline(mode)

        qa_chain = qa_chain_cache[mode]
        result = qa_chain.invoke({"query": query})
        if isinstance(result, dict):
            return result.get("answer") or result.get("result") or "No answer found."
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

