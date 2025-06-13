import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_documents(data_dir="../data/documents"):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def main():
    documents = get_documents()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_texts(documents, embedding=embedding_model, persist_directory="../data/chroma_db")
    db.persist()
    print("Vector DB created and saved!")

if __name__ == "__main__":
    main()