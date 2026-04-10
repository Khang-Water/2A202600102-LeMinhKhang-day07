# Metric Evaluation - Lab 7 (docs/EVALUATION.md)

## 1. Scope va setup da chay

- Data set dung de danh gia:
  - `data/XanhSM - User FAQs.md`
  - `data/XanhSM - electric_motor_driver FAQs.md`
  - `data/XanhSM - electric_car_driver FAQs.md`
  - `data/XanhSM - Restaurant FAQs.md`
- Embedding backend: `_mock_embed` (mac dinh cua project, chay offline).
- Chunking strategies da so sanh:
  - `FixedSizeChunker(chunk_size=220, overlap=40)`
  - `SentenceChunker(max_sentences_per_chunk=3)`
  - `RecursiveChunker(chunk_size=260)`

## 2. 5 benchmark queries + gold answers

| # | Query | Gold answer tom tat |
|---|---|---|
| 1 | Huong dan yeu cau xuat hoa don VAT va cach kiem tra hoa don voi cac chuyen xe Xanh SM | Can yeu cau truoc khi chuyen di ket thuc; co the tao yeu cau tren app hoac goi 1900 2097; hoa don gui qua email va xem lai trong lich su chuyen di. |
| 2 | Lam sao khi hanh khach de quen do tren xe? | Tai xe cung cap thong tin cho CSKH de khach lien he nhan lai; neu chua xac dinh duoc chu do thi ghi chu va mang den Trung tam ho tro. |
| 3 | Ngoai luong thuong, toi con duoc huong chinh sach gi nua? | Co kham suc khoe dinh ky, BHXH khi ky hop dong chinh thuc, dao tao, tu van lo trinh nghe nghiep va phuc loi khac. |
| 4 | Toi muon dat chuyen giao do an tren ung dung | Dat tren app theo quy trinh chon dich vu, diem lay/giao, hinh thuc thanh toan, xac nhan dat. |
| 5 | Quan co rating tren Google va muon dong bo ve Ung dung Xanh SM | Dong bo dinh ky khi ten + dia chi trung khop va rating Google >= 4.0; neu du dieu kien ma chua hien thi thi gui yeu cau ho tro. |

## 3. Metric #1 - Retrieval Precision

### 3.1 Top-k relevance va score distribution theo strategy

| Strategy | Queries co >=2 ket qua relevant trong top-3 | Avg top-1 score |
|---|---:|---:|
| fixed_size_220_40 | 0/5 | 0.374 |
| sentence_3 | 0/5 | 0.334 |
| recursive_260 | 0/5 | 0.358 |

Nhan xet:
- Score top-1 nam trong khoang gan nhau (`~0.33` den `~0.37`), khong tach bach ro relevant/nhieu.
- Top-3 relevance thap tren ca 3 strategy.

## 4. Metric #2 - Chunk Coherence

Ket qua `ChunkingStrategyComparator.compare(..., chunk_size=220)`:

| Tai lieu | fixed_size (count/avg_len) | by_sentences (count/avg_len) | recursive (count/avg_len) |
|---|---|---|---|
| XanhSM - User FAQs | 191 / 219.1 | 112 / 333.2 | 278 / 134.0 |
| XanhSM - electric_car_driver FAQs | 15 / 207.4 | 12 / 230.2 | 19 / 145.7 |
| XanhSM - Restaurant FAQs | 99 / 218.6 | 55 / 351.1 | 143 / 134.7 |

Nhan xet:
- `by_sentences` tao chunk dai nhat (coherence ngon ngu tot hon, nhung co nguy co qua dai).
- `recursive` tao nhieu chunk ngan hon (tot cho retrieve chi tiet, nhung de mat context neu query can nhieu thong tin lien tuc).

## 5. Metric #3 - Metadata Utility (A/B)

A/B test `search()` vs `search_with_filter()` voi `RecursiveChunker(260)`:

| Query | Target category | Unfiltered: top-3 dung category | Filtered: top-3 dung category |
|---|---|---:|---:|
| Q1 VAT invoice | user | 2/3 | 3/3 |
| Q2 lost item | bike_driver | 0/3 | 3/3 |
| Q3 benefits | car_driver | 1/3 | 3/3 |
| Q4 food delivery booking | user | 2/3 | 3/3 |
| Q5 Google rating sync | restaurant | 0/3 | 3/3 |

Nhan xet:
- Metadata filter tang precision rat ro (match category top-3 dat 3/3 cho 5/5 query).
- Trade-off: score top-1 co query giam (vi khong con "nhay" sang category khac co score random cao hon).

## 6. Metric #4 - Grounding Quality

Mau kiem tra `KnowledgeBaseAgent.answer()` cho Q1, Q3 cho thay:
- Agent co dua cau tra loi tren context duoc retrieve.
- Nhung do retrieval ban dau chua trung y chinh, cau tra loi bi grounded vao context "sai phan" (grounded but not correct).

Ket luan grounding:
- Prompt grounding trong `KnowledgeBaseAgent` la dung.
- Nut that nam o retrieval quality dau vao.

## 7. Metric #5 - Data Strategy Impact

Ket qua chi ra:
- Bo tai lieu co nhieu domain con (user, bike_driver, car_driver, restaurant), nen metadata schema (`category`, `language`, `doc_id`, `source`) co tac dung lon.
- Neu khong filter, retrieval de bi "lac domain".
- Khi filter dung domain, ket qua top-k on dinh hon va de trace source hon.

## 8. Failure analysis (bat buoc)

Failure case tieu bieu: Q2 "hanh khach de quen do".
- Unfiltered top-3 = 0/3 dung category, top-1 lay doan khong lien quan.
- Nguyen nhan:
  - Embedding backend dang dung `_mock_embed` (deterministic random vector, khong mang y nghia semantic that).
  - Tai lieu dai, nhieu chu de, nen random similarity de chon sai.
- Cai thien de xac:
  1. Dung real embedding backend (`LocalEmbedder` hoac `OpenAIEmbedder`).
  2. Bat buoc metadata pre-filter theo `category` truoc khi search.
  3. Chunk theo cau/section + metadata chi tiet hon (`section`, `faq_id`) de tang precision.

## 9. Kiem tra yeu cau metric da hoan thanh

- [x] Retrieval Precision
- [x] Chunk Coherence
- [x] Metadata Utility (A/B filtered vs unfiltered)
- [x] Grounding Quality
- [x] Data Strategy Impact
- [x] Co failure analysis va de xuat cai thien

