import os
import warnings
import sys
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Additional Imports ---
from mistralai import Mistral
from mistralai import UserMessage, SystemMessage, AssistantMessage
from langchain_mistralai.embeddings import MistralAIEmbeddings
from PyPDF2 import PdfReader, errors
from langchain_community.document_loaders import PyMuPDFLoader
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi  # Not used in retrieval_only, but keep for potential hybrid search later
from collections import Counter
import torch
from langchain_community.vectorstores import Milvus
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
# from langchain.retrievers import EnsembleRetriever  # Not needed in this simplified version
from langchain_community.cache import InMemoryCache, SQLiteCache
# from langchain_community.retrievers import BM25Retriever  # Not used in retrieval_only
from langchain_core.documents import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not found in .env file")

client = Mistral(api_key=api_key)
MODEL = "mistral-large-latest"

cache_dir = "./.cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = cache_dir
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
set_llm_cache(InMemoryCache())
set_llm_cache(SQLiteCache(database_path="./.langchain.db"))  # Good practice to keep
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading all NLTK data. This may take a while...")
    nltk.download('all')
    print("NLTK data downloaded.")


# --- Helper Functions --- (Keep these, they are useful)
def load_documents_from_pdfs(pdf_dir: str) -> List[Document]:
    """Loads documents from PDFs using Langchain's PyMuPDFLoader."""
    documents = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_single_pdf, os.path.join(pdf_dir, filename)): filename
            for filename in os.listdir(pdf_dir)
            if filename.lower().endswith(".pdf")
        }
        for future in as_completed(futures):
            try:
                documents.extend(future.result())
            except Exception as e:
                print(f"Error loading PDF: {futures[future]}: {e}")
    return documents

def load_single_pdf(filepath: str) -> List[Document]:
    """Loads a single PDF using PyMuPDFLoader."""
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    filename = os.path.basename(filepath)
    for doc in docs:
        doc.metadata['filename'] = filename
        if 'file_path' in doc.metadata:
            del doc.metadata['file_path']
        if 'source' in doc.metadata:
            del doc.metadata['source']
    return docs

# --- Embedding --- (Keep this)
class LLMEmbedder:
    def __init__(self, model_name: str = "mistral-embed", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedder = MistralAIEmbeddings(model=model_name, mistral_api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            embeddings = self.embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

# --- Reranker --- (You might not need this in the retrieval_only version, but it's good to have)
class MistralReranker:
    def __init__(self, model_name: str = "mistral-large-latest", batch_size: int = 4):
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        scores = []
        doc_texts = [doc.page_content for doc in docs]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_batch, query, doc_texts[i:i + self.batch_size])
                       for i in range(0, len(doc_texts), self.batch_size)]
            for future in as_completed(futures):
                scores.extend(future.result())
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]

    def process_batch(self, query: str, batch: List[str]) -> List[float]:
        batch_scores = []
        messages_batch = [
            [
                SystemMessage(
                    content="You are a helpful assistant that ranks documents based on their relevance to a query. Rank the following document as a float between 0 and 1, with 1 being the most relevant.",
                ),
                UserMessage(
                    content=f"Query: {query}\nDocument: {doc_text}\nRank:",
                ),
            ]
            for doc_text in batch
        ]
        try:
            for messages in messages_batch:
                response = self.client.chat(model=self.model_name, messages=messages)
                try:
                    score = float(response.choices[0].message.content)
                    batch_scores.append(score)
                except ValueError:
                    print(f"Warning: Could not parse ranking score. Defaulting to 0.")
                    batch_scores.append(0.0)
        except Exception as e:
            print(f"An error occurred during reranking: {e}")
            batch_scores.extend([0.0] * len(batch))
        return batch_scores


# --- Semantic Chunking --- (Keep this, but you only need it if you re-index)
def semantic_chunk_texts(documents: List[Document],
                         embedder,
                         similarity_threshold: float = 0.7,
                         target_chunk_size: int = 512,
                         chunk_size_flexibility: float = 0.2,
                         window_size: int = 5,
                         overlap: int = 2,
                         min_chunk_sentences: int = 3) -> List[Document]:

    all_chunks = []
    lower_bound = int(target_chunk_size * (1 - chunk_size_flexibility))
    upper_bound = int(target_chunk_size * (1 + chunk_size_flexibility))

    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        if not sentences:
            continue
        sentence_embeddings = embedder.embed(sentences)

        current_chunk: List[str] = []
        current_chunk_embeddings: List[List[float]] = []

        for i in range(0, len(sentences), window_size - overlap):
            window_sentences = sentences[i:i + window_size]
            window_embeddings = sentence_embeddings[i:i + window_size]

            if not current_chunk:
                current_chunk.extend(window_sentences)
                current_chunk_embeddings.extend(window_embeddings)
            else:
                avg_current_embedding = np.mean(current_chunk_embeddings, axis=0)
                avg_window_embedding = np.mean(window_embeddings, axis=0)
                similarity = cosine_similarity([avg_current_embedding], [avg_window_embedding])[0][0]
                current_chunk_tokens = sum(len(s.split()) for s in current_chunk)
                window_tokens = sum(len(s.split()) for s in window_sentences)

                if (similarity >= similarity_threshold and
                        lower_bound <= current_chunk_tokens + window_tokens <= upper_bound):
                    current_chunk.extend(window_sentences[-(window_size - overlap):])
                    current_chunk_embeddings.extend(window_embeddings[-(window_size - overlap):])
                else:
                    if len(current_chunk) >= min_chunk_sentences:
                        new_doc = Document(page_content=" ".join(current_chunk), metadata=doc.metadata.copy())
                        all_chunks.append(new_doc)
                    current_chunk = window_sentences
                    current_chunk_embeddings = window_embeddings
        if current_chunk:
            new_doc = Document(page_content=" ".join(current_chunk), metadata=doc.metadata.copy())
            all_chunks.append(new_doc)

    return all_chunks



# Modified retrieval_only_pipeline function
def retrieval_only_pipeline(initial_query: str, collection_name: str = "rag_collection") -> Dict[str, Any]:
    """
    Performs retrieval-based question answering using an existing Milvus collection.
    Does NOT handle document loading or indexing.  Starts an interactive chat session.

    Args:
        initial_query:  The user's first question or statement.  Used to seed the retrieval.
        collection_name: The name of the Milvus collection to query.

    Returns:
        A dictionary containing the chat history.  The "answer" key is not used, as the
        output is the interactive chat.
    """
    from pymilvus import connections, Collection, utility

    print("=== 1) Connecting to Milvus ===")
    connections.connect("default", host="localhost", port="19530")

    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist in Milvus.")

    vector_store = Collection(collection_name)  # No schema needed - it already exists
    vector_store.load()  # Load the collection into memory

    print("\n=== 2) Embedding Query ===")
    embedder = LLMEmbedder()  # Create embedder instance
    query_embedding = embedder.embed([initial_query])[0]

    print("\n=== 3) Retrieval from Milvus ===")
    search_params = {
        "metric_type": "L2",  # Consistent metric
        "params": {"nprobe": 10},
    }

    result = vector_store.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=5,  # Retrieve top 5
        expr=None,
        output_fields=["text", "filename"],
    )

    retrieved_docs = []
    for hits in result:
        for hit in hits:
            retrieved_docs.append(
                Document(  # Use Document objects, for consistency
                    page_content=hit.entity.get("text"),
                    metadata={"filename": hit.entity.get("filename")},
                )
            )

    print("\n=== 4) Summarization and QA (Mistral Large) ===")

    # --- CRITICAL CHANGE:  Build context from retrieved documents ---
    context = " ".join([doc.page_content for doc in retrieved_docs])
    if not context:
        context = "No relevant documents found."  # Handle case with no results
    else:
        context = "Relevant context from your documents:\n" + context

    # Create initial messages, including the context
    initial_messages = [
        SystemMessage(content=f"""
            Tu es une messagerie vocale d'hôpital. Tu vas classer les messages en cinq catégories, allant de "très urgent" à "moins urgent". Voici comment je vais procéder :

        1. **Très urgent** : Situations nécessitant une intervention immédiate (ex. : crise cardiaque, difficulté respiratoire sévère).
        2. **Urgent** : Problèmes médicaux sérieux nécessitant une attention rapide (ex. : douleur intense, blessure grave).
        3. **Modéré** : Problèmes de santé nécessitant une attention, mais pas immédiate (ex. : fièvre, douleur modérée).
        4. **Léger** : Problèmes mineurs pouvant attendre (ex. : rhume, éruption cutanée légère).
        5. **Non urgent** : Questions générales ou administratives (ex. : rendez-vous, informations).

        Vous pouvez maintenant m'envoyer des messages, et je les classerai en conséquence.

        dans chaques cas essaye d'approfondir pour mieux comprendre les symptomes

        D'accord, je vais essayer d'approfondir chaque cas pour mieux comprendre les symptômes. Voici quelques questions et considérations pour chaque niveau de gravité :

        1. **Très urgent** :
        - **Crise cardiaque** : Douleur thoracique intense, essoufflement, sueurs, nausées.
            - *Questions* : La douleur irradie-t-elle vers le bras ou la mâchoire ? Avez-vous des antécédents de maladies cardiaques ?
        - **Difficulté respiratoire sévère** : Respiration rapide et superficielle, sensation d'étouffement.
            - *Questions* : Avez-vous de l'asthme ou une autre condition respiratoire ? La difficulté est-elle apparue soudainement ?

        2. **Urgent** :
        - **Douleur intense** : Douleur aiguë qui empêche les activités normales.
            - *Questions* : Où est localisée la douleur ? Est-elle constante ou intermittente ?
        - **Blessure grave** : Fractures, coupures profondes.
            - *Questions* : La blessure saigne-t-elle abondamment ? Pouvez-vous bouger la partie affectée ?

        3. **Modéré** :
        - **Fièvre** : Température corporelle élevée.
            - *Questions* : Quelle est votre température exacte ? Avez-vous d'autres symptômes comme des frissons ou des maux de tête ?
        - **Douleur abdominale** : Inconfort ou douleur dans l'abdomen.
            - *Questions* : La douleur est-elle localisée ou généralisée ? Avez-vous des nausées ou des vomissements ?

        4. **Léger** :
        - **Rhume** : Congestion nasale, éternuements.
            - *Questions* : Depuis combien de temps avez-vous ces symptômes ? Avez-vous de la fièvre ?
        - **Éruption cutanée légère** : Rougeurs ou boutons sur la peau.
            - *Questions* : L'éruption est-elle douloureuse ou démange ? Avez-vous été en contact avec des allergènes ?

        5. **Non urgent** :
        - **Rendez-vous** : Planification ou modification de rendez-vous.
            - *Questions* : Quel type de rendez-vous souhaitez-vous ? Avez-vous des préférences de date ou d'heure ?
        - **Informations** : Questions générales sur les services de l'hôpital.
            - *Questions* : Quel service ou département vous intéresse ? Avez-vous besoin d'informations spécifiques ?

        Ces questions peuvent aider à mieux comprendre la situation et à fournir des conseils plus appropriés. Si vous avez un autre message ou symptôme à partager, n'hésitez pas !

        pose les questions une a une a chaque fois comme si nous étions en conversation et que tu attendais une réponse de ma part pour continuer

            Utilise ce contexte pour ta réponse, mais priorise la question posée par l'utilisateur:

            {context}
            """),
        UserMessage(content=f"{initial_query}"),  # User's first message/query
    ]

    # Initialize chat_history with the initial messages
    chat_history = initial_messages.copy()

    # --- Interactive Chat Loop ---
    while True:
        try:
            # Choose the correct method to call based on the client object structure
            if hasattr(client, 'chat') and callable(client.chat):
                response = client.chat(model=MODEL, messages=chat_history)
            elif hasattr(client, 'chat') and hasattr(client.chat, 'complete'):
                response = client.chat.complete(model=MODEL, messages=chat_history)
            else:
                raise AttributeError("Could not find a valid way to call the Mistral client")

            assistant_response = response.choices[0].message.content
            print(f"Assistant: {assistant_response}")
            chat_history.append(AssistantMessage(content=assistant_response))

            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "stop", "bye"]:
                break
            chat_history.append(UserMessage(content=user_input))

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return {
        "answer": '',  # No direct answer in chat mode
        "source_documents": [doc.metadata['filename'] for doc in retrieved_docs],  # List of filenames
        "chat_history": [message.model_dump() if hasattr(message, 'model_dump') else {'role': message.role, 'content': message.content} for message in chat_history]
    }



# --- Example Usage ---
if __name__ == "__main__":
    pdf_directory = "Doc"
    results = retrieval_only_pipeline("Start")  # Use retrieval_only_pipeline

    # The chat happens within the retrieval_only_pipeline function.
    # You don't need to print results here; the conversation is interactive.