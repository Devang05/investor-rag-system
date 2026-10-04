
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
        chunk_size = 120,
        chunk_overlap = 20,
        length_function = len
    )

    #making chunk
    chunks = splitter.create_documents(
        texts = [doc["page_content"]],
        metadatas = [doc["metadata"]]
    )

    return chunks

#defining vector store
def vector_store():

    # defining embedding model 
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device":"cpu"}
    )
    
    vector_store = Chroma(
        collection_name="Documents_DB",
        persist_directory="./chroma_db",
        embedding_function = embedding_model,
        collection_configuration = {"hnsw":{"space":"cosine"}}
    )

    return vector_store

def update_document(vector_store,document):

    # lsit of all chunks with same doc_id
    existing = vector_store.get(
        where={"doc_id":document["metadata"]["doc_id"]}
    )

    old_ids = existing["ids"]

    # make chunks to update db
    chunks = make_chunks(document)

    if not chunks:
        return "Document Empty"

    # chunk_ids
    chunk_ids = []
    for i in range(len(chunks)):
        # sample chunk_id -> demo_rules_1_<context_hash>_chunk 1, 2, etc.
        chunk_ids.append(f"{document['metadata']['doc_id']}_{document['metadata']['content_hash']}_chunk{i}")
    
    new_ids = chunk_ids

    # checking new_ids and old_ids is same -> unchanged
    if set(old_ids)==set(new_ids):
        return "Unchanged Data Base"

    # adding new chunks to database
    vector_store.add_documents(
        documents = chunks,
        ids = new_ids
    )

    #removing old/obsolete chunk_ids
    obsolete_chunk_ids = set(old_ids)-set(new_ids)
    if obsolete_chunk_ids:
        vector_store.delete(ids=list(obsolete_chunk_ids))

    if old_ids:
        return "Data Base Updated"

    return "New Document Added"

if __name__ == "__main__":
    db = vector_store()
    document = load_doc("./DATA_FILES/04_demo_tax_rules.txt","SYN_TAX_01")
    print(update_document(db, document))