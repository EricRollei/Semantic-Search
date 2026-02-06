# Quick Reference Card

## Basic Workflows

### 🔍 Text Search
```
LoadEmbeddingModel → LoadOrCreateIndex → SearchByText → PreviewResults
```

### 🖼️ Image Search
```
LoadEmbeddingModel → LoadOrCreateIndex → SearchByImage → PreviewResults
```

### 📹 Video Search
```
LoadEmbeddingModel → LoadOrCreateIndex → SearchByVideo → PreviewResults
```

### 📄 Document Search
```
LoadEmbeddingModel → LoadOrCreateIndex → SearchByDocument → PreviewResults
```

### 🚫 Search with Exclusion
```
LoadEmbeddingModel → LoadOrCreateIndex → SearchWithExclusion → PreviewResults
                                         (query + exclude terms)
```

### 🔀 Multi-Index Search
```
LoadEmbeddingModel ──┬──→ SearchMultiIndex → PreviewResults
LoadOrCreateIndex 1 ─┤    (searches all)
LoadOrCreateIndex 2 ─┤
LoadOrCreateIndex 3 ─┘
```

### 🎯 Two-Stage Search (Best Accuracy)
```
LoadEmbeddingModel ──┬──→ SearchByText (top_k=50) ──→ RerankResults (top_k=10)
LoadRerankerModel ───┘                                        │
LoadOrCreateIndex ────────────────────────────────────────────┘
```

### 📁 Index New Folder
```
LoadEmbeddingModel → LoadOrCreateIndex → AddFolderToIndex → GetIndexInfo
```

---

## Node Quick Reference

| Node | Purpose | Key Parameters |
|------|---------|----------------|
| **Load Embedding Model** | Load encoder | `max_resolution`, `attention_type` |
| **Load Reranker Model** | Load reranker | `max_resolution`, `attention_type` |
| **Load/Create Index** | Open/create index | `index_name`, `index_type` |
| **Rebuild Index** | Convert index type | `target_type` |
| **Compact Index** | Reclaim deleted space | `new_index_type` |
| **Add Folder to Index** | Index media | `folder_path`, `recursive`, `include_videos`, `include_documents` |
| **Search by Text** | Text query | `query`, `top_k`, `min_score` |
| **Search by Image** | Image query | `image`/`image_path`, `top_k`, `min_score` |
| **Search by Video** | Video query | `video_path`, `top_k`, `max_frames` |
| **Search by Document** | PDF query | `pdf_path`, `page_number`, `top_k` |
| **Search with Exclusion** | Exclude terms | `query`, `exclude`, `exclusion_threshold` |
| **Search Multi-Index** | Cross-index | `index_1-4`, `normalize_scores` |
| **Rerank Results** | Improve ranking | `top_k`, `min_score` |
| **Filter by Score** | Remove low scores | `min_score`, `max_results` |
| **Preview Results** | Show grid | `columns`, `thumbnail_size`, `show_scores` |
| **Get Result Paths** | Extract paths | `include_scores` |

---

## FAISS Index Types

| Type | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| `Flat (Exact)` | Slowest | 100% | <10K images |
| `IVF-Flat (Fast)` | Fast | ~95-99% | 10K-100K images |
| `HNSW (Very Fast)` | Fastest | ~95-99% | 10K-1M images |

**Note**: IVF-Flat needs training on 1000+ vectors before use.

---

## Recommended Settings

### For Indexing (Accuracy Priority)
- Resolution: `1024x1024 (1MP)` or higher
- Model: 8B
- Batch size: 8 (adjust for VRAM)

### For Search (Speed Priority)
- Resolution: `512x512` or `768x768`
- Use reranking for final precision

### Score Thresholds
| Stage | Recommended min_score |
|-------|----------------------|
| Initial search | 0.10 - 0.20 |
| After reranking | 0.25 - 0.40 |

---

## Attention Types

| Type | Speed | Compatibility |
|------|-------|---------------|
| `sdpa` | Fast | Default, works everywhere |
| `eager` | Medium | Fallback option |
| `sage` | Fastest | Requires SageAttention installed |

---

## Example Queries

**Portraits:**
- "woman with red hair in vintage dress"
- "dramatic portrait with rim lighting"
- "studio headshot on gray background"

**Landscapes:**
- "sunset over mountains with orange sky"
- "misty forest in morning light"
- "beach with waves at golden hour"

**Products:**
- "minimalist product on white background"
- "luxury watch closeup"
- "food photography with natural light"

**Styles:**
- "film noir aesthetic high contrast"
- "pastel colors soft dreamy"
- "cyberpunk neon city"

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Lower batch_size, use 2B model |
| Slow indexing | Check GPU usage, use sage attention |
| Poor accuracy | Increase resolution, use 8B + reranking |
| Version warning | `pip install qwen-vl-utils==0.0.14` |

---

## Supported Formats

**Images**: jpg, jpeg, png, webp, bmp, tiff, gif, heic, heif, raw (cr2, cr3, nef, arw, dng, etc.)

**Videos**: mp4, mkv, avi, mov, webm, wmv, flv, m4v

**Documents**: pdf (each page indexed separately)

---

## File Locations

```
H:/semantic_search/
├── models/           # Downloaded models
│   ├── Qwen3-VL-Embedding-8B/
│   └── Qwen3-VL-Reranker-8B/
└── indexes/          # Your indexes
    └── my_index/
        ├── faiss.index
        ├── metadata.db
        └── thumbnails/
```
