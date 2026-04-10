# Bao Cao Lab 7: Embedding va Vector Store

**Ho ten:** [Dien ten ban]  
**Nhom:** [Dien ten nhom]  
**Ngay nop:** 2026-04-10

---

## 1. Warm-up (5 diem)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghia la gi?**  
Hai vector embedding co huong gan nhau, nghia la hai doan van co nghia tuong dong (du tu vung co the khac).

**Vi du HIGH similarity**
- Sentence A: Python is widely used in machine learning.
- Sentence B: Python is commonly applied in AI and ML tasks.
- Tai sao tuong dong: Ca hai deu noi ve vai tro cua Python trong AI/ML.

**Vi du LOW similarity**
- Sentence A: Python is widely used in machine learning.
- Sentence B: I like playing football on weekends.
- Tai sao khac: Hai cau thuoc hai chu de khong lien quan.

**Tai sao cosine similarity uu tien hon Euclidean distance cho text embeddings?**  
Cosine similarity tap trung vao huong vector (semantic direction), it bi anh huong boi do lon vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ky tu, chunk_size=500, overlap=50. Bao nhieu chunks?**
- Formula: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
- Tinh: `ceil((10000 - 50) / (500 - 50)) = ceil(9950/450) = 23`
- **Dap an: 23 chunks**

**Neu overlap tang len 100 thi sao?**
- `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900/400) = 25`
- Overlap cao hon giup giu context giua cac chunk nhung tang so chunk va chi phi index/retrieval.

---

## 2. Document Selection - Nhom (10 diem)

### Domain & Ly do chon

**Domain:** Xanh SM FAQ (user + driver + merchant/restaurant).  

**Tai sao chon domain nay?**  
Bo FAQ co nhieu nhom doi tuong va nhieu quy trinh nghiep vu, rat phu hop de kiem tra retrieval precision, metadata filtering va grounding.

### Data Inventory

| # | Ten tai lieu | Nguon | So ky tu | Metadata da gan |
|---|---|---|---:|---|
| 1 | XanhSM - User FAQs.md | Internal dataset | 50196 | category=user, language=vi, source |
| 2 | XanhSM - electric_motor_driver FAQs.md | Internal dataset | 11662 | category=bike_driver, language=vi, source |
| 3 | XanhSM - electric_car_driver FAQs.md | Internal dataset | 3583 | category=car_driver, language=vi, source |
| 4 | XanhSM - Restaurant FAQs.md | Internal dataset | 25352 | category=restaurant, language=vi, source |
| 5 | XanhSM - FAQs.md | Internal dataset | 50196 | category=user_general, language=vi, source |

### Metadata Schema

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|---|---|---|---|
| doc_id | string | XanhSM - User FAQs | Ho tro delete theo tai lieu goc |
| category | string | user / bike_driver / car_driver / restaurant | Ho tro pre-filter theo domain cau hoi |
| language | string | vi | Tranh mix ket qua khac ngon ngu |
| source | string | data/XanhSM - User FAQs.md | Truy vet nguon chunk va doi chieu |

---

## 3. Chunking Strategy - Ca nhan chon, nhom so sanh (15 diem)

### Baseline Analysis

Chay `ChunkingStrategyComparator().compare(..., chunk_size=220)`:

| Tai lieu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|---|---|---:|---:|---|
| XanhSM - User FAQs.md | FixedSizeChunker (`fixed_size`) | 191 | 219.1 | Trung binh |
| XanhSM - User FAQs.md | SentenceChunker (`by_sentences`) | 112 | 333.2 | Tot nhat theo cau |
| XanhSM - User FAQs.md | RecursiveChunker (`recursive`) | 278 | 134.0 | Tot theo section nho |
| XanhSM - Restaurant FAQs.md | FixedSizeChunker (`fixed_size`) | 99 | 218.6 | Trung binh |
| XanhSM - Restaurant FAQs.md | SentenceChunker (`by_sentences`) | 55 | 351.1 | Tot nhat theo cau |
| XanhSM - Restaurant FAQs.md | RecursiveChunker (`recursive`) | 143 | 134.7 | Tot cho retrieve theo heading |

### Strategy cua toi

**Loai:** `RecursiveChunker(chunk_size=260)` + metadata pre-filter theo `category`.  

**Mo ta cach hoat dong:**  
Recursive chunking tach theo thu tu separator tu lon den nho (doan, dong, cau, tu, ky tu), giup giu heading/section khi co the va van dam bao gioi han kich thuoc chunk.

**Tai sao toi chon strategy nay cho domain nhom?**  
FAQ co cau truc heading + noi dung ngan theo muc. Recursive giu duoc heading QA trong chunk, va khi ket hop metadata filter thi precision on dinh hon unfiltered retrieval.

### So sanh: Strategy cua toi vs baseline

OpenAI embedding model su dung: `text-embedding-3-large`.

| Strategy | Queries co >=2 relevant chunks trong top-3 | Avg top-1 score |
|---|---:|---:|
| FixedSizeChunker(220,40) | 1/5 | 0.5769 |
| SentenceChunker(3) | 2/5 | 0.5399 |
| **RecursiveChunker(260)** | **1/5** | **0.6020** |

Ghi chu: Recursive co top-1 score cao nhat, nhung theo metric strict `>=2 relevant in top-3` thi Sentence nhinh hon.

### So sanh voi thanh vien khac

| Thanh vien | Strategy | Retrieval Score (/10) | Diem manh | Diem yeu |
|---|---|---|---|---|
| Toi | Recursive + metadata filter | Dang cap nhat | Score top-1 cao, trace source ro | De lac y neu khong pre-filter |
| [Ten 1] | [Can bo sung] | [Can bo sung] | [Can bo sung] | [Can bo sung] |
| [Ten 2] | [Can bo sung] | [Can bo sung] | [Can bo sung] | [Can bo sung] |

**Strategy nao tot nhat cho domain nay? Tai sao?**  
Neu khong dung metadata filter, ca 3 strategy deu co query bi lac domain. Khi bat metadata filter, retrieval dung domain tang ro ret (nhieu query tu 0/3 len 3/3 dung category).

---

## 4. My Approach - Ca nhan (10 diem)

### Chunking Functions

**`SentenceChunker.chunk` - approach:**  
Dung regex tach ranh gioi cau, loai bo khoang trang rong, sau do gom theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split` - approach:**  
Split de quy theo danh sach separator uu tien; neu khong tach duoc thi fallback hard-slice theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search` - approach:**  
Index embeddings theo chunk, luu metadata va id record. Khi search: embed query, tinh score va sap xep giam dan.

**`search_with_filter` + `delete_document` - approach:**  
Filter metadata truoc search de tang precision theo domain. `delete_document` xoa toan bo records co `doc_id` trung.

### KnowledgeBaseAgent

**`answer` - approach:**  
Lay top-k chunks tu store, inject vao prompt theo block `[Chunk i | score]`, yeu cau LLM chi tra loi dua tren context.

### Test Results

```text
pytest tests/ -v
...
======================== 42 passed, 1 warning in 0.30s ========================
```

**So tests pass:** 42 / 42

---

## 5. Similarity Predictions - Ca nhan (5 diem)

| Pair | Sentence A | Sentence B | Du doan | Actual Score | Dung? |
|---|---|---|---|---|---|
| 1 | Python for ML | Python for AI tasks | high | high | Co |
| 2 | RAG combines retrieval + generation | Vector DB stores embeddings | medium | medium | Co |
| 3 | Cooking pasta | Database indexing | low | low | Co |
| 4 | Chunk overlap preserves context | Overlap keeps neighboring meaning | high | high | Co |
| 5 | Soccer training | Prompt engineering | low | low | Co |

**Ket qua nao bat ngo nhat?**  
Cap 2 co lexical overlap khong cao nhung van gan nghia o muc vua, cho thay embedding model bat semantic relation tot hon keyword matching.

---

## 6. Results - Ca nhan (10 diem)

Chay benchmark voi OpenAI embedding model `text-embedding-3-large`.

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|---|---|
| 1 | Huong dan yeu cau xuat hoa don VAT va cach kiem tra hoa don voi cac chuyen xe Xanh SM | Yeu cau truoc khi chuyen di ket thuc; app/hotline 1900 2097; hoa don gui email va xem lai trong lich su chuyen di. |
| 2 | Lam sao khi hanh khach de quen do tren xe? | Cung cap thong tin cho CSKH de khach lien he nhan lai do; neu khong xac dinh duoc chu do thi ghi chu va mang den trung tam ho tro. |
| 3 | Ngoai luong thuong, toi con duoc huong chinh sach gi nua? | Kham suc khoe dinh ky, BHXH, dao tao, tu van lo trinh nghe nghiep, phuc loi khac. |
| 4 | Toi muon dat chuyen giao do an tren ung dung | Chon dich vu giao hang tren app, nhap diem nhan/giao, chon thanh toan va xac nhan dat. |
| 5 | Quan co rating tren Google va muon dong bo ve Ung dung Xanh SM | Dong bo dinh ky khi ten+dia chi trung va rating >= 4.0; neu chua hien thi thi gui yeu cau ho tro. |

### Ket Qua Cua Toi (config: RecursiveChunker + metadata filter)

| # | Query | Top-1 Retrieved Chunk (tom tat) | Score | Relevant? | Agent Answer (tom tat) |
|---|---|---|---:|---|---|
| 1 | VAT invoice | Chunk heading dung section 3.4 ve xuat hoa don VAT + hotline/email | 0.7853 | Yes | Tra loi dung quy trinh xuat hoa don |
| 2 | Lost item | Chunk heading dung section 2.1 ve hanh khach de quen do | 0.5439 | Yes | Tra loi dung huong xu ly co CSKH |
| 3 | Driver benefits | Chunk heading dung section 1.2 ve chinh sach ngoai luong thuong | 0.3535 | Yes (heading-level) | Tra loi can bo sung chi tiet phuc loi |
| 4 | Food delivery booking | Top-1 roi vao section ngoai app-trip, top-2 moi la section dat giao hang | 0.4538 | Partial | Co nguy co thieu/lech buoc dat don |
| 5 | Google rating sync | Chunk heading dung section 7.4 + dieu kien rating >=4.0 | 0.8072 | Yes | Tra loi dung dieu kien dong bo |

**Bao nhieu queries tra ve chunk relevant trong top-3?** 2 / 5 (theo metric strict `>=2 relevant chunks in top-3`)

### Metadata Utility (A/B)

| Query | Target category | Unfiltered match top-3 | Filtered match top-3 |
|---|---|---:|---:|
| Q1 | user | 3/3 | 3/3 |
| Q2 | bike_driver | 0/3 | 3/3 |
| Q3 | car_driver | 0/3 | 3/3 |
| Q4 | user | 2/3 | 3/3 |
| Q5 | restaurant | 3/3 | 3/3 |

Ket luan: metadata pre-filter co tac dong lon nhat den precision domain.

---

## 7. What I Learned (5 diem - Demo)

**Dieu hay nhat toi hoc duoc tu thanh vien khac trong nhom:**  
Khi chunking strategy giong nhau nhung metadata schema khac, ket qua retrieval da thay doi ro. Metadata khong chi la phan phu ma la "retrieval control".

**Dieu hay nhat toi hoc duoc tu nhom khac (qua demo):**  
Nhom khac dung heading-aware chunking cho FAQ giup reduce noise o top-k. Vi vay domain co cau truc nen uu tien chunk theo heading/section thay vi fixed-size thuần.

**Neu lam lai, toi se thay doi gi trong data strategy?**  
Them metadata `section`, `faq_id`, `audience` va viet query router nho (map query -> category truoc search). Dieu nay se giam failure case nhu Q4.

### Failure Analysis (bat buoc)

Failure case: Q4 "dat chuyen giao do an tren ung dung".  
Top-1 sau filter van chua la chunk dat giao hang chuan (top-2 moi dung section), dan den nguy co answer chua day du.

Nguyen nhan:
1. Query mo ho giua "giao do an" va "chuyen xe ngoai ung dung".  
2. Chunk co heading gan nghia nhung context khac nghiep vu.

Cai thien:
1. Query expansion (them tu khoa `giao hang`, `diem nhan`, `diem giao`).  
2. Rerank top-k bang lexical + semantic hybrid.  
3. Bo sung metadata `service_type` de filter sat hon.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|---|---|---:|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 9 / 10 |
| Chunking strategy | Nhom | 13 / 15 |
| My approach | Ca nhan | 10 / 10 |
| Similarity predictions | Ca nhan | 5 / 5 |
| Results | Ca nhan | 9 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 4 / 5 |
| **Tong** |  | **85 / 100** |

