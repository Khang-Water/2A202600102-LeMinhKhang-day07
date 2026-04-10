# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Minh Khang
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, thể hiện rằng hai đoạn văn có ý nghĩa ngữ nghĩa tương đồng, dù cách diễn đạt có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Python is widely used in machine learning.
- Sentence B: Python is commonly applied in AI and ML tasks.
- Tại sao tương đồng:
> Cả hai câu đều nói về việc Python được dùng trong AI/ML, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: Python is widely used in machine learning.
- Sentence B: I like playing football on weekends.
- Tại sao khác:
> Hai câu thuộc hai domain hoàn toàn khác nhau, không có liên hệ ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo hướng của vector thay vì độ dài, nên phản ánh tốt hơn ý nghĩa ngữ nghĩa của văn bản. Euclidean distance bị ảnh hưởng bởi magnitude nên không phù hợp với embeddings.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> num_chunks = ceil((10000 - 50) / (500 - 50))  
> = ceil(9950 / 450)  
> ≈ ceil(22.11)  
> = 23

**Đáp án:** 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> chunk count tăng lên (~25 chunks) vì step nhỏ hơn. Overlap lớn giúp giữ context giữa các chunk, cải thiện chất lượng retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Internal AI & Technical Documentation

**Tại sao nhóm chọn domain này?**
> Nhóm chọn tài liệu kỹ thuật và AI vì các tài liệu này có cấu trúc rõ ràng (đoạn, section, guideline), phù hợp để đánh giá hiệu quả của các chiến lược chunking và retrieval trong hệ thống RAG.

---

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | internal | ~2000 | category=python |
| 2 | rag_system_design.md | internal | ~3000 | category=rag |
| 3 | vector_store_notes.md | internal | ~2000 | category=vector |
| 4 | customer_support_playbook.txt | internal | ~2000 | category=support |
| 5 | vi_retrieval_notes.md | internal | ~2500 | category=retrieval |

---

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "rag", "python" | lọc theo domain |
| language | string | "en", "vi" | tránh trộn ngôn ngữ |
| source | string | "internal_doc" | truy vết nguồn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| sample | FixedSizeChunker | ~20 | ~500 | ❌ |
| sample | SentenceChunker | ~15 | không đều | ⚠️ |
| sample | RecursiveChunker | ~18 | cân bằng | ✅ |

---

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo thứ tự ưu tiên: đoạn → dòng → câu → từ → ký tự. Nếu một đoạn vẫn vượt quá chunk_size, thuật toán sẽ đệ quy xuống separator nhỏ hơn cho đến khi đảm bảo kích thước phù hợp.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu kỹ thuật có cấu trúc rõ ràng theo đoạn và section, nên recursive chunking tận dụng tốt cấu trúc này để giữ ngữ nghĩa và cải thiện retrieval quality.

---

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| sample | FixedSize | ~20 | đều | thấp |
| sample | **Recursive (của tôi)** | ~18 | cân bằng | cao |

---

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Minh Khang (Tôi) | Recursive | 9 | giữ context tốt | phức tạp hơn |
| Thế Anh| Recursive (250 chars) | 8.752 | Trích xuất chính xác, duy trì được thông tin quan trọng | Số chunk nhiều, dẫn đến dư thừa dữ liệu do overlap |
| Dương Khoa Điềm | RecursiveChunker  | 7.9 | Giữ được ngữ cảnh cụm Q&A tương đối ổn. | Tuỳ biến sai sót separator khiến một số câu dài bị đứt vụn, điểm chưa cao. |
| Võ Thanh Chung        | RecursiveChunker (250 chars) | 8                     | Giữ cấu trúc tự nhiên, chunk đều | Có thể cắt ngang câu dài |
| Nguyễn Hồ Bảo Thiên | FixedSizeChunker (chunk_size=100, overlap=20) | 8.56 | Xử lý nhanh | Dễ ngắt câu giữa chừng, gây mất ngữ nghĩa |
| Tuyền | Recursive (350 chars) | 8.77 | Giữ context, Q&A coherent, score cao nhất | Số chunk nhiều (654), tốn memory |

---

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker tốt nhất vì cân bằng giữa context và chunk size, giúp retrieval chính xác hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**SentenceChunker.chunk — approach:**
> Sử dụng regex `(?<=[.!?])\s+` để tách câu. Sau đó normalize whitespace và group các câu thành chunk theo số lượng cố định.

**RecursiveChunker.chunk / _split — approach:**
> Thuật toán sử dụng đệ quy. Base case là khi text nhỏ hơn chunk_size. Nếu không, split theo separator hiện tại, nếu vẫn quá lớn thì gọi lại với separator nhỏ hơn.

---

### EmbeddingStore

**add_documents + search — approach:**
> Mỗi document được embed thành vector và lưu vào store. Search sử dụng dot product giữa query embedding và stored embeddings để xếp hạng.

**search_with_filter + delete_document — approach:**
> search_with_filter lọc metadata trước rồi mới search. delete_document loại bỏ tất cả chunk có cùng document id.

---

### KnowledgeBaseAgent

**answer — approach:**
> Agent retrieve top-k chunks, ghép thành context, sau đó tạo prompt và gọi LLM để trả lời dựa trên context.

---

### Test Results
![alt text](image.png)


**Số tests pass:** 42/42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python for ML | Python for AI | high | ~0.9 | ✓ |
| 2 | Python | Football | low | ~0.1 | ✓ |
| 3 | RAG system | retrieval system | high | ~0.85 | ✓ |
| 4 | Cooking | Database | low | ~0.2 | ✓ |
| 5 | ML | DL | high | ~0.9 | ✓ |

---

**Kết quả nào bất ngờ nhất?**
> Một số câu không cùng từ khóa vẫn có similarity cao, cho thấy embedding hiểu semantic chứ không chỉ keyword.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Python dùng để làm gì? | dùng cho AI, backend |
| 2 | RAG là gì? | retrieval + generation |
| 3 | vector store là gì? | lưu embeddings |
| 4 | chunking ảnh hưởng gì? | ảnh hưởng retrieval |
| 5 | metadata có tác dụng gì? | filter và improve precision |

---

### Kết Quả Của Tôi

| # | Query | Top-1 Chunk | Score | Relevant? | Answer |
|---|-------|------------|-------|-----------|--------|
| 1 | Python | python doc | 0.9 | ✓ | đúng |
| 2 | RAG | rag doc | 0.88 | ✓ | đúng |
| 3 | vector store | vector doc | 0.87 | ✓ | đúng |
| 4 | chunking | retrieval doc | 0.8 | ✓ | đúng |
| 5 | metadata | vector doc | 0.78 | ✓ | đúng |

---

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Custom chunking theo structure (Q&A hoặc section) có thể outperform generic chunking.

**Điều hay nhất tôi học được từ nhóm khác:**
> Metadata filtering giúp cải thiện precision đáng kể trong retrieval.

**Nếu làm lại, tôi sẽ thay đổi gì?**
> Tôi sẽ tối ưu chunk_size, thêm overlap và thiết kế metadata tốt hơn để cải thiện retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 |
| Document selection | Nhóm | 10 |
| Chunking strategy | Nhóm | 15 |
| My approach | Cá nhân | 10 |
| Similarity predictions | Cá nhân | 5 |
| Results | Cá nhân | 10 |
| Core implementation | Cá nhân | 30 |
| Demo | Nhóm | 5 |
| **Tổng** | | **90-100** |
