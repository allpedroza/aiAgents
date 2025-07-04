import os
import numpy as np
import faiss
import json
import pickle
import google.generativeai as gen_ai
from google import genai
from typing import TypedDict, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Portuguese stopwords for TF-IDF vectorizer
PORTUGUESE_STOPWORDS = [
    # Artigos
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "as", "às", "da", "das", "de", "dela", 
    "delas", "dele", "deles", "do", "dos", "duas", "esta", "estas", "este", "estes", "esta", "está", "estás", 
    "o", "os", "um", "uma", "umas", "uns",
    
    # Preposições/contrações
    "com", "como", "contra", "desde", "em", "entre", "para", "perante", "por", "sem", "sob", "sobre", "trás",
    "pela", "pelas", "pelo", "pelos", "num", "numa", "nuns", "numas", "dum", "duma", "duns", "dumas",
    
    # Pronomes
    "ele", "eles", "eu", "lhe", "lhes", "me", "meu", "meus", "minha", "minhas", "nós", "se", "seu", "seus", 
    "sua", "suas", "te", "tu", "tua", "tuas", "você", "vocês", "vos",
    
    # Conjunções
    "e", "mas", "nem", "ou", "porém", "que", "quer", "se", "então", "todavia",
    
    # Advérbios/interjeições
    "agora", "aí", "ainda", "ali", "amanhã", "antes", "aqui", "assim", "bem", "cedo", "depois", "hoje", 
    "logo", "mais", "mal", "melhor", "menos", "muito", "não", "onde", "ontem", "pra", "qual", "quando", 
    "quanto", "quê", "sim", "talvez", "tão", "tarde", "tem", "têm", "sim", "já", "só", "talvez",
    
    # Outros
    "etc", "exemplo", "isso", "isto", "outro", "outros", "qualquer", "seja", "também", "outra", "ser", "há",
    "outras", "tempo", "vez", "vezes", "via"
]

def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms

# Global singleton vector database - implemented exactly as in original code
class GeminiDocumentProcessor:
    _instance = None

    @staticmethod
    def get_instance(MODEL_NAME=None, CACHE_DIR=None):
        """Get or create the singleton instance"""
        if GeminiDocumentProcessor._instance is None:
            if MODEL_NAME is None or CACHE_DIR is None:
                raise ValueError("Parâmetros obrigatórios na primeira chamada: MODEL_NAME e CACHE_DIR")
            GeminiDocumentProcessor._instance = GeminiDocumentProcessor(MODEL_NAME, CACHE_DIR)
        return GeminiDocumentProcessor._instance

    def __init__(self, MODEL_NAME, CACHE_DIR):
        """Initialize the processor (called only once)"""
        print("Inicializando processador de documentos...")

        self.model_name = MODEL_NAME
        self.cache_dir = CACHE_DIR

        print("Carregando modelo de embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)
        self.document_chunks = []
        self.document_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_size = 2000

        self.alpha = 0.7
        self.beta = 0.2
        self.gamma = 0.1

        self.model = gen_ai.GenerativeModel(self.model_name)

        self.load_state()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files"""
        print(f"Extraindo texto de {pdf_path} (PDF)...")
        try:
            from PyPDF2 import PdfReader
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                return text
        except Exception as e:
            print(f"Erro ao processar o PDF: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT files"""
        print(f"Extraindo texto de {txt_path} (TXT)...")
        try:
            with open(txt_path, 'r', encoding="utf-8") as file:
                text = file.read()
            return text
        except Exception as e:
            print(f"Erro ao processar o arquivo TXT: {e}")
            return ""
    
    def extract_chunks_from_preformatted_txt(self, txt_path: str) -> List[str]:
        """Extract chunks from preformatted TXT files"""
        print(f"Lendo chunks pré-formatados de {txt_path}...")
        try:
            with open(txt_path, 'r', encoding="utf-8") as file:
                text = file.read()
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            print(f"Arquivo {txt_path} lido com {len(chunks)} chunk(s).")
            return chunks
        except Exception as e:
            print(f"Erro ao ler o arquivo TXT: {e}")
            return []
    
    def extract_text_from_json(self, json_path: str) -> str:
        """Extract text from JSON files"""
        print(f"Extraindo texto de {json_path} (JSON)...")
        try:
            with open(json_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
            text = json.dumps(data, indent=4, ensure_ascii=False)
            return text
        except Exception as e:
            print(f"Erro ao processar o arquivo JSON: {e}")
            return ""
    
    def extract_text_from_csv(self, csv_path: str) -> List[str]:
        """Extract text from CSV files"""
        print(f"Extraindo e agrupando metadados de {csv_path} (CSV)...")
        table_data = {}
        try:
            with open(csv_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    table_id = f"{row['table_catalog']}.{row['table_schema']}.{row['table_name']}"
                    if table_id not in table_data:
                        table_data[table_id] = {
                            "table_catalog": row["table_catalog"],
                            "table_schema": row["table_schema"],
                            "table_name": row["table_name"],
                            "columns": []
                        }
                    table_data[table_id]["columns"].append({
                        "column_name": row["column_name"],
                        "data_type": row["data_type"],
                        "description": row["description"]
                    })
        except Exception as e:
            print(f"Erro ao processar o CSV: {e}")
            return []

        chunks = []
        for table_id, info in table_data.items():
            lines = []
            lines.append(f"Tabela: {info['table_catalog']}.{info['table_schema']}.{info['table_name']}")
            lines.append("Colunas:")
            for col in info["columns"]:
                lines.append(f"- {col['column_name']} ({col['data_type']}): {col['description']}")
            chunks.append("\n".join(lines))
        print(f"Extração concluída: {len(chunks)} chunk(s) gerado(s).")
        return chunks

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                print(f"Chunk {len(chunks)} - tamanho: {len(chunk_text)} caracteres")
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            print(f"Chunk {len(chunks)} - tamanho: {len(chunk_text)} caracteres")
            
        print(f"Texto dividido em {len(chunks)} chunks")
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        print(f"Gerando embeddings para {len(texts)} chunk(s)...")
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            print(f"Erro ao gerar embeddings: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def _build_lexical_index(self, texts: List[str]) -> None:
        """Build lexical index with TF-IDF"""
        print("Construindo índice lexical (TF-IDF)...")
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=PORTUGUESE_STOPWORDS)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def _calculate_importance_scores(self, texts: List[str]) -> np.ndarray:
        """Calculate importance scores based on text length"""
        importance = np.array([len(text.split()) for text in texts])
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance
    
    def index_document(self, file_path: str) -> None:
        """Index document from file path"""
        file_path = file_path.strip()
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
            chunks = self.split_text_into_chunks(text)
        elif file_path.lower().endswith('.json'):
            text = self.extract_text_from_json(file_path)
            chunks = self.split_text_into_chunks(text)
        elif file_path.lower().endswith('.csv'):
            chunks = self.extract_text_from_csv(file_path)
        elif file_path.lower().endswith('.txt'):
            if os.path.basename(file_path) == "resultados_4.txt" or os.path.basename(file_path) == "resultados_5.txt":
                chunks = self.extract_chunks_from_preformatted_txt(file_path)
            else:
                text = self.extract_text_from_txt(file_path)
                chunks = self.split_text_into_chunks(text)
        else:
            print("Tipo de arquivo não suportado. Utilize PDF, TXT, JSON ou CSV.")
            return
        
        if not chunks:
            print("Não foi possível extrair conteúdo do documento.")
            return
        
        self.document_chunks = chunks
        
        embeddings = self._generate_embeddings(chunks)
        normalized_embeddings = _normalize_embeddings(embeddings)
        self.document_embeddings = normalized_embeddings.copy()
        
        self._build_lexical_index(chunks)
        self.importance_scores = self._calculate_importance_scores(chunks)
        self.index.reset()
        self.index.add(normalized_embeddings.astype(np.float32))
        print(f"Documento indexado com sucesso! {len(chunks)} chunk(s) adicionados ao índice.")
        self._save_state()
    
    def _save_state(self):       
        chunks_filename = os.path.join(self.cache_dir, "document_chunks.json")
        index_filename = os.path.join(self.cache_dir, "faiss_index.bin")
        tfidf_filename = os.path.join(self.cache_dir, "tfidf_model.pkl")
        importance_filename = os.path.join(self.cache_dir, "importance_scores.npy")
 
        print(f'AQUI!!{chunks_filename}')
        """Save state to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            with open(chunks_filename, "w", encoding="utf-8") as f:
                json.dump(self.document_chunks, f, ensure_ascii=False)
            faiss.write_index(self.index, index_filename)
            
            if self.tfidf_vectorizer is not None:
                with open(tfidf_filename, "wb") as f:
                    pickle.dump(self.tfidf_vectorizer, f)
            
            if hasattr(self, 'importance_scores'):
                np.save(importance_filename, self.importance_scores)
                
            print("Estado salvo com sucesso!")
        except Exception as e:
            print(f"Erro ao salvar estado: {e}")
    
    def load_state(self) -> bool:
        chunks_filename=os.path.join(self.cache_dir, "document_chunks.json") 
        index_filename=os.path.join(self.cache_dir, "faiss_index.bin")
        tfidf_filename=os.path.join(self.cache_dir, "tfidf_model.pkl") 
        importance_filename=os.path.join(self.cache_dir, "importance_scores.npy")
        
        """Load state from disk"""
        try:
            if os.path.exists(chunks_filename):
                with open(chunks_filename, "r", encoding="utf-8") as f:
                    self.document_chunks = json.load(f)
                    
            if os.path.exists(index_filename):
                self.index = faiss.read_index(index_filename)
                
                if self.document_chunks:
                    embeddings = self._generate_embeddings(self.document_chunks)
                    self.document_embeddings = _normalize_embeddings(embeddings)
                    
                    if os.path.exists(tfidf_filename):
                        with open(tfidf_filename, "rb") as f:
                            self.tfidf_vectorizer = pickle.load(f)
                        self.tfidf_matrix = self.tfidf_vectorizer.transform(self.document_chunks)
                    else:
                        self._build_lexical_index(self.document_chunks)
                        
                    if os.path.exists(importance_filename):
                        self.importance_scores = np.load(importance_filename)
                    else:
                        self.importance_scores = self._calculate_importance_scores(self.document_chunks)
                        
                print(f"Estado carregado com sucesso! {len(self.document_chunks)} chunk(s) disponíveis.")
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar estado: {e}")
            return False

    def clear_memory(self):
        
        chunks_filename=os.path.join(self.cache_dir, "document_chunks.json")
        index_filename=os.path.join(self.cache_dir, "faiss_index.bin")
        tfidf_filename=os.path.join(self.cache_dir, "tfidf_model.pkl") 
        importance_filename=os.path.join(self.cache_dir, "importance_scores.npy")
                    
        """Clear memory and state files"""
        self.document_chunks = []
        self.document_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.importance_scores = None
        self.index = faiss.IndexFlatIP(self.dimension)
        
        for filename in [chunks_filename, index_filename, tfidf_filename, importance_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                
        print("Memória limpa com sucesso!")
    
    def dartboard_ranking(self, query: str, top_k_initial: int = 30) -> List[Dict[str, Any]]:
        """Rank chunks using Dartboard algorithm"""
        if self.index.ntotal == 0:
            print("Nenhum documento indexado.")
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            normalized_query = _normalize_embeddings(query_embedding)
            distances, indices = self.index.search(normalized_query.astype(np.float32), top_k_initial)
            
            if self.tfidf_vectorizer is not None:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                lexical_scores = np.zeros(len(indices[0]))
                for i, idx in enumerate(indices[0]):
                    lexical_scores[i] = cosine_similarity(
                        query_tfidf, self.tfidf_matrix[idx].reshape(1, -1)
                    ).item()
            else:
                lexical_scores = np.zeros(len(indices[0]))
            
            importance_scores = np.zeros(len(indices[0]))
            if hasattr(self, 'importance_scores'):
                for i, idx in enumerate(indices[0]):
                    importance_scores[i] = self.importance_scores[idx]
            
            combined_scores = (
                self.alpha * distances[0] +
                self.beta * lexical_scores +
                self.gamma * importance_scores
            )
            
            reranked_indices = np.argsort(-combined_scores)
            results = []
            for i in reranked_indices:
                doc_idx = indices[0][i]
                results.append({
                    "chunk_id": int(doc_idx),
                    "score": float(combined_scores[i]),
                    "text": self.document_chunks[doc_idx],
                    "semantic_score": float(distances[0][i]),
                    "lexical_score": float(lexical_scores[i]),
                    "importance_score": float(importance_scores[i])
                })
            
            return results
        except Exception as e:
            print(f"Erro no ranking Dartboard: {e}")
            return []
    
    def _filter_diverse_results(self, results: List[Dict[str, Any]], diversity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Filter results for diversity"""
        selected = []
        selected_ids = []
        
        for result in results:
            chunk_id = result["chunk_id"]
            candidate_embedding = self.document_embeddings[chunk_id]
            is_redundant = False
            for s in selected_ids:
                sim = np.dot(candidate_embedding, self.document_embeddings[s])
                if sim > diversity_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                selected.append(result)
                selected_ids.append(chunk_id)
        return selected

    def query_document(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query documents using Dartboard ranking with diversity filtering"""
        if self.index.ntotal == 0:
            print("Nenhum documento indexado ainda.")
            return []
            
        try:
            retrieval_k = max(top_k * 3, 30)
            initial_results = self.dartboard_ranking(query, retrieval_k)
            diverse_results = self._filter_diverse_results(initial_results, diversity_threshold=0.95)
            if len(diverse_results) > top_k:
                diverse_results = diverse_results[:top_k]
            return diverse_results
        except Exception as e:
            print(f"Erro ao consultar documento: {e}")
            return []
    
    def adjust_dartboard_weights(self, alpha: float = None, beta: float = None, gamma: float = None):
        """Adjust Dartboard weights"""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        print(f"Pesos Dartboard ajustados: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
