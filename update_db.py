
import argparse
import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# chroma_db is kept next to this script, regardless of the folder it is run from
DB_DIR = Path(__file__).resolve().parent / "chroma_db"

# chunk settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Reading file with doc_id
def load_doc(file_path,doc_id):
    with open (file_path,"r",encoding="utf-8")as file:
        text = file.read()

    # content_hash for new file check - hashlib
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return{
        "page_content":text,
        "metadata":{
            "source":file_path,
            "doc_id":doc_id,
            "content_hash":content_hash
        }
    }

# defining chunking/text splitter
def make_chunks(doc):

    #defining splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len
    )

    #making chunk
    chunks = splitter.create_documents(
        texts = [doc["page_content"]],
        metadatas = [doc["metadata"]]
    )

    return chunks

#defining vector store
def get_vector_store():

    # defining embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device":"cpu"}
    )

    store = Chroma(
        collection_name="Documents_DB",
        persist_directory=str(DB_DIR),
        embedding_function = embedding_model,
        collection_configuration = {"hnsw":{"space":"cosine"}}
    )

    return store

def update_document(db,document):

    # lsit of all chunks with same doc_id
    existing = db.get(
        where={"doc_id":document["metadata"]["doc_id"]}
    )

    old_ids = existing["ids"]

    # make chunks to update db
    chunks = make_chunks(document)

    # empty document -> remove its old chunks so they are no longer searchable
    if not chunks:
        if old_ids:
            db.delete(ids=old_ids)
            return "Document Empty - Old Chunks Removed"
        return "Document Empty"

    # chunk_ids
    chunk_ids = []
    for i in range(len(chunks)):
        # sample chunk_id -> SYN_TAX_01_<content_hash>_chunk0, chunk1, etc.
        chunk_ids.append(f"{document['metadata']['doc_id']}_{document['metadata']['content_hash']}_chunk{i}")

    new_ids = chunk_ids

    # checking new_ids and old_ids is same -> unchanged
    if set(old_ids)==set(new_ids):
        return "Unchanged Data Base"

    # adding new chunks to database
    db.add_documents(
        documents = chunks,
        ids = new_ids
    )

    #removing old/obsolete chunk_ids
    obsolete_chunk_ids = set(old_ids)-set(new_ids)
    if obsolete_chunk_ids:
        db.delete(ids=list(obsolete_chunk_ids))

    if old_ids:
        return "Data Base Updated"

    return "New Document Added"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add or update a document in the vector database.")
    parser.add_argument("file_path", help="path to the text file, e.g. DATA_FILES/04_demo_tax_rules.txt")
    parser.add_argument("doc_id", help="stable document ID, e.g. SYN_TAX_01 (reuse it to update the same document)")
    args = parser.parse_args()

    db = get_vector_store()
    document = load_doc(args.file_path, args.doc_id)
    print(update_document(db, document))
