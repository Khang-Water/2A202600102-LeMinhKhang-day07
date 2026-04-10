# RAG Pipeline Giai Thich File-to-File

Tai lieu nay mo ta toan bo pipeline cua project theo dung thu tu chay, tu file dau vao den cau tra loi cuoi.

## 1) Ban do tong quan

Pipeline RAG trong repo nay co 2 luong:

1. Indexing pipeline (nap tri thuc vao vector store)
2. Query pipeline (tim context + tao cau tra loi)

So do nhanh:

```
main.py
  -> load file .md/.txt thanh Document
  -> chon embedding backend (mock/local/openai/github_models)
  -> EmbeddingStore.add_documents(...)
      -> embedding text
      -> luu vao in-memory hoac ChromaDB
  -> KnowledgeBaseAgent.answer(question)
      -> store.search(question)
      -> tao prompt tu top-k chunks
      -> goi llm_fn(prompt)
```

## 2) Vai tro tung file trong src

### `src/models.py`
- Dinh nghia `Document` dataclass:
  - `id`: dinh danh tai lieu
  - `content`: noi dung text
  - `metadata`: thong tin bo sung (`source`, `extension`, `doc_id`, ...)
- Day la "don vi du lieu" truyen giua cac thanh phan.

### `src/embeddings.py`
- Cung cap 3 backend embedding:
  - `MockEmbedder`: deterministic, khong can internet/API key
  - `LocalEmbedder`: dung `sentence-transformers`
  - `OpenAIEmbedder`: goi OpenAI-compatible embeddings API
- `_mock_embed = MockEmbedder()` la fallback an toan de test/lab.
- Cac env constants giup chon backend:
  - `EMBEDDING_PROVIDER`
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`
  - `GITHUB_MODELS_TOKEN`, ...

### `src/chunking.py`
- Chua cac chunking strategies va similarity:
  - `FixedSizeChunker`: cat theo kich thuoc co overlap
  - `SentenceChunker`: gom toi da N cau/chunk
  - `RecursiveChunker`: split theo uu tien separator
  - `compute_similarity`: cosine similarity
  - `ChunkingStrategyComparator`: so sanh 3 strategy
- Trong pipeline chinh hien tai, `EmbeddingStore` luu theo `Document.content`.
  Chunking module la "hop cong cu" de ban ap dung khi muon index theo chunk nho.

### `src/store.py`
- `EmbeddingStore` la trung tam retrieval:
  - `add_documents(docs)`: embed + luu
  - `search(query, top_k)`: embed query + cham diem + tra top-k
  - `search_with_filter(...)`: loc metadata truoc roi moi search
  - `delete_document(doc_id)`: xoa toan bo chunks/doc theo `doc_id`
  - `get_collection_size()`: so luong records
- Ho tro 2 che do:
  - ChromaDB neu import duoc
  - In-memory fallback neu khong co ChromaDB

### `src/agent.py`
- `KnowledgeBaseAgent` implement RAG answer loop:
  1. retrieve top-k tu `EmbeddingStore`
  2. xep context thanh cac block `[Chunk i | score=...]`
  3. tao prompt "chi duoc tra loi dua tren context"
  4. goi `llm_fn(prompt)` de sinh answer

### `src/__init__.py`
- Export public API de file ngoai import gon:
  - `Document`, `EmbeddingStore`, `KnowledgeBaseAgent`
  - chunkers, embedders, constants

## 3) Luong chay file-to-file trong `main.py`

## Buoc A: Nap du lieu dau vao
- `load_documents_from_files(...)` doc cac file `.md/.txt`.
- Moi file hop le duoc doi thanh 1 `Document`.
- Metadata mac dinh:
  - `source`: duong dan file
  - `extension`: `.md` hoac `.txt`

## Buoc B: Chon embedding backend
- `run_manual_demo(...)` doc `.env` qua `load_dotenv`.
- Chon backend theo `EMBEDDING_PROVIDER`:
  - `local` -> `LocalEmbedder`
  - `openai` -> `OpenAIEmbedder`
  - `github`/`github_models` -> `OpenAIEmbedder` voi base URL GitHub Models
  - khac/loi -> fallback `_mock_embed`

## Buoc C: Indexing vao store
- Tao `EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)`.
- Goi `store.add_documents(docs)`:
  1. tao id record (`{doc.id}_{index}`)
  2. dam bao metadata co `doc_id`
  3. embed document content
  4. luu vao ChromaDB hoac `_store` in-memory

## Buoc D: Retrieval test
- Goi `store.search(query, top_k=3)`.
- Store se:
  1. embed query
  2. tinh score voi tung record
     - in-memory: dot product (do embeddings da normalized, gan cosine)
     - chroma: doc `distances` va doi dau thanh score am de sap xep giam dan
  3. tra danh sach ket qua da sort

## Buoc E: Agent answer (RAG)
- Tao `agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)`.
- Goi `agent.answer(query, top_k=3)`:
  1. retrieve lai top-k chunks
  2. build prompt co question + context
  3. goi `demo_llm(prompt)` -> tra ve answer

Luu y: Trong `main.py`, `demo_llm` chi la mock LLM de minh hoa.

## 4) Minh hoa de hieu bang vi du nho

Gia su co 2 tai lieu:

1. `python_intro.txt`: "Python la ngon ngu lap trinh..."
2. `vector_store_notes.md`: "Vector store dung de tim van ban tuong tu..."

Query: `Vector store dung de lam gi?`

Luong xu ly:

1. `main.py` doc file -> tao 2 `Document`.
2. `EmbeddingStore.add_documents` embed 2 noi dung va luu lai.
3. Query duoc embed thanh vector.
4. So sanh query vector voi 2 vector da luu.
5. Tai lieu lien quan hon (`vector_store_notes`) co score cao hon.
6. `KnowledgeBaseAgent` lap prompt:
   - Context chunk 1: noi dung vector store
   - Context chunk 2: noi dung python intro (neu top_k=2)
7. `llm_fn` doc prompt va sinh cau tra loi dua tren context.

## 5) Mapping RAG theo dung thanh phan ky thuat

1. Retriever
   - `EmbeddingStore.search` / `search_with_filter`
2. Knowledge base
   - du lieu da index trong ChromaDB hoac `_store`
3. Generator
   - `llm_fn` truyen vao `KnowledgeBaseAgent`
4. Prompt assembly
   - `KnowledgeBaseAgent.answer` tao context blocks + instruction

## 6) Noi can nang cap neu muon "RAG that"

1. Them chunking truoc khi add vao store:
   - Dung `SentenceChunker`/`RecursiveChunker`
   - Moi chunk thanh 1 `Document` con (metadata giu `doc_id`, `chunk_id`)
2. Dung metadata filter nhieu hon:
   - vi du `{"category": "policy", "language": "vi"}`
3. Thay `demo_llm` bang LLM that
4. Them danh gia benchmark:
   - top-k relevance, grounding quality, failure analysis

## 7) Test lien quan de xac nhan pipeline

- `tests/test_solution.py` bao phu:
  - chunkers
  - similarity
  - store add/search/filter/delete
  - agent answer non-empty

Neu can nhanh: chay `pytest tests -v`.

