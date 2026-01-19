# Polymath v4 - Corpus Gap Analysis

**Generated:** 2026-01-18
**Current State:** 1,701 documents, 1,791 repositories

---

## Executive Summary

**Major Gaps Identified:**

| Category | Status | Priority |
|----------|--------|----------|
| Foundational Tool Papers (Scanpy, Seurat, etc.) | ❌ MISSING | 🔴 Critical |
| H&E → Gene Expression Methods | ⚠️ Partial | 🔴 Critical |
| Pathology Foundation Models | ⚠️ Partial | 🔴 Critical |
| Spatial Deconvolution Methods | ⚠️ Partial | 🟡 High |
| Foundational DL Papers | ❌ MISSING | 🟡 High |
| Spatial Platform Papers (Visium, Xenium, etc.) | ⚠️ Partial | 🟡 High |

**Paradox:** We have excellent repo coverage (Scanpy, Seurat, Cell2location repos indexed) but are missing the papers that describe these tools.

---

## Category 1: Foundational Tool Papers 🔴 CRITICAL

We have the GitHub repos but NOT the seminal papers describing them.

| Tool | Paper | DOI | Status |
|------|-------|-----|--------|
| Scanpy | Wolf et al. (2018) Genome Biology | 10.1186/s13059-017-1382-0 | ❌ Missing |
| Seurat v3 | Stuart et al. (2019) Cell | 10.1016/j.cell.2019.05.031 | ❌ Missing |
| Seurat v4 | Hao et al. (2021) Cell | 10.1016/j.cell.2021.04.048 | ❌ Missing |
| Seurat v5 | Hao et al. (2024) Nature Biotechnology | 10.1038/s41587-023-01767-y | ❌ Missing |
| Cell2location | Kleshchevnikov et al. (2022) Nat Biotech | 10.1038/s41587-021-01139-4 | ❌ Missing |
| Squidpy | Palla et al. (2022) Nature Methods | 10.1038/s41592-021-01358-2 | ❌ Missing |
| SpatialData | Marconato et al. (2024) Nature Methods | 10.1038/s41592-024-02212-x | ❌ Missing |
| Cellpose | Stringer et al. (2021) Nature Methods | 10.1038/s41592-020-01018-x | ❌ Missing |
| Cellpose 2.0 | Pachitariu & Stringer (2022) Nat Methods | 10.1038/s41592-022-01663-4 | ❌ Missing |
| StarDist | Schmidt et al. (2018) MICCAI | 10.1007/978-3-030-00934-2_30 | ❌ Missing |
| AnnData | Virshup et al. (2024) Nature Biotechnology | 10.1038/s41587-023-01733-8 | ❌ Missing |
| scvi-tools | Gayoso et al. (2022) Nature Biotechnology | 10.1038/s41587-021-01206-w | ❌ Missing |
| Tangram | Biancalani et al. (2021) Nature Methods | 10.1038/s41592-021-01264-7 | ❌ Missing |

---

## Category 2: H&E → Gene Expression Methods 🔴 CRITICAL

**Directly relevant for Img2ST research.**

| Method | Paper | DOI | Status |
|--------|-------|-----|--------|
| HE2RNA | Schmauch et al. (2020) Nat Commun | 10.1038/s41467-020-17678-4 | ❌ Missing |
| ST-Net | He et al. (2020) Nature Commun | 10.1038/s41467-020-20249-2 | ❌ Missing |
| BLEEP | Xie et al. (2024) Nature Methods | 10.1038/s41592-024-02284-9 | ❌ Missing |
| iStar | Zhang et al. (2024) Nature Genetics | 10.1038/s41588-024-01773-9 | ❌ Missing |
| THItoGene | Jia et al. (2024) arXiv | 10.48550/arXiv.2408.01435 | ❌ Missing |
| Hist2ST | Zeng et al. (2022) npj Precis Onc | 10.1038/s41698-022-00315-2 | ❌ Missing |
| EGN | Yang et al. (2023) Nature Machine Intell | 10.1038/s42256-023-00717-6 | ❌ Missing |
| HEST-1k | Jaume et al. (2024) NeurIPS | 10.48550/arXiv.2406.16192 | ✅ Have |
| STimage-1K4M | 2024 | 10.48550/arXiv.2406.06393 | ✅ Have |
| Img2ST-Net | Our paper (2025) | 10.1117/1.JMI.12.6.061410 | ✅ Have |

---

## Category 3: Pathology Foundation Models 🔴 CRITICAL

| Model | Paper | DOI | Status |
|-------|-------|-----|--------|
| HIPT | Chen et al. (2022) CVPR | - | ❌ Missing |
| CTransPath | Wang et al. (2022) Med Image Anal | 10.1016/j.media.2022.102559 | ❌ Missing |
| UNI | Chen et al. (2024) Nature Medicine | 10.1038/s41591-024-02857-3 | ❌ Missing |
| Virchow | Vorontsov et al. (2024) Nature Medicine | 10.1038/s41591-024-02856-4 | ⚠️ Partial |
| GigaPath | Xu et al. (2024) Nature | 10.1038/s41586-024-07441-w | ❌ Missing |
| Phikon | Filiot et al. (2023) arXiv | 10.48550/arXiv.2309.07778 | ❌ Missing |
| PLIP | Huang et al. (2023) Nature Medicine | 10.1038/s41591-023-02504-3 | ❌ Missing |
| CONCH | Lu et al. (2024) Nature Medicine | 10.1038/s41591-024-02886-6 | ❌ Missing |
| Hibou | Nechaev et al. (2024) arXiv | 10.48550/arXiv.2406.02896 | ❌ Missing |

---

## Category 4: Spatial Deconvolution Methods 🟡 HIGH

| Method | Paper | DOI | Status |
|--------|-------|-----|--------|
| RCTD | Cable et al. (2022) Nature Biotechnology | 10.1038/s41587-021-00830-w | ❌ Missing |
| SPOTlight | Elosua-Bayes et al. (2021) Nucleic Acids Res | 10.1093/nar/gkab043 | ❌ Missing |
| CARD | Ma & Zhou (2022) Nature Biotechnology | 10.1038/s41587-022-01273-7 | ❌ Missing |
| DestVI | Lopez et al. (2022) Nature Biotechnology | 10.1038/s41587-022-01272-8 | ✅ Have |
| SpatialDWLS | Dong & Yuan (2021) Genome Biology | 10.1186/s13059-021-02362-7 | ❌ Missing |
| stereoscope | Andersson et al. (2020) Communications Bio | 10.1038/s42003-020-01247-y | ❌ Missing |
| BayesPrism | Chu et al. (2022) Nature Cancer | 10.1038/s43018-022-00356-3 | ❌ Missing |

---

## Category 5: Foundational Deep Learning Papers 🟡 HIGH

| Paper | Citation | DOI | Status |
|-------|----------|-----|--------|
| Attention Is All You Need | Vaswani et al. (2017) NeurIPS | 10.48550/arXiv.1706.03762 | ❌ Missing |
| U-Net | Ronneberger et al. (2015) MICCAI | 10.1007/978-3-319-24574-4_28 | ❌ Missing |
| ResNet | He et al. (2016) CVPR | 10.1109/CVPR.2016.90 | ❌ Missing |
| Vision Transformer (ViT) | Dosovitskiy et al. (2021) ICLR | 10.48550/arXiv.2010.11929 | ❌ Missing |
| DINO | Caron et al. (2021) ICCV | 10.48550/arXiv.2104.14294 | ❌ Missing |
| DINOv2 | Oquab et al. (2023) | 10.48550/arXiv.2304.07193 | ❌ Missing |
| MAE | He et al. (2022) CVPR | 10.48550/arXiv.2111.06377 | ❌ Missing |
| CLIP | Radford et al. (2021) | 10.48550/arXiv.2103.00020 | ❌ Missing |
| SimCLR | Chen et al. (2020) ICML | 10.48550/arXiv.2002.05709 | ❌ Missing |

---

## Category 6: Spatial Transcriptomics Platform Papers 🟡 HIGH

| Platform | Paper | DOI | Status |
|----------|-------|-----|--------|
| Visium (10x) | Stahl et al. (2016) Science | 10.1126/science.aaf2403 | ❌ Missing |
| Visium HD | 10x Genomics (2023) | N/A (whitepaper) | ❌ Missing |
| Slide-seq | Rodriques et al. (2019) Science | 10.1126/science.aaw1219 | ❌ Missing |
| Slide-seqV2 | Stickels et al. (2021) Nature Biotechnology | 10.1038/s41587-020-0739-1 | ❌ Missing |
| MERFISH | Chen et al. (2015) Science | 10.1126/science.aaa6090 | ❌ Missing |
| Xenium | Janesick et al. (2023) bioRxiv | 10.1101/2022.10.06.510405 | ❌ Missing |
| seqFISH+ | Eng et al. (2019) Nature | 10.1038/s41586-019-1049-y | ❌ Missing |
| CosMx | He et al. (2022) Nature Biotechnology | 10.1038/s41587-022-01483-z | ❌ Missing |
| CODEX | Goltsev et al. (2018) Cell | 10.1016/j.cell.2018.07.010 | ❌ Missing |
| STARmap | Wang et al. (2018) Science | 10.1126/science.aat5691 | ❌ Missing |

---

## Category 7: Single-Cell Genomics Landmark Papers 🟢 MEDIUM

| Paper | Citation | DOI | Status |
|-------|----------|-----|--------|
| Drop-seq | Macosko et al. (2015) Cell | 10.1016/j.cell.2015.05.002 | ❌ Missing |
| 10x Chromium | Zheng et al. (2017) Nature Commun | 10.1038/ncomms14049 | ❌ Missing |
| CellRanger | 10x Genomics | N/A | ❌ Missing |
| scRNA-seq normalization | Hafemeister & Satija (2019) Genome Bio | 10.1186/s13059-019-1874-1 | ❌ Missing |
| Harmony | Korsunsky et al. (2019) Nature Methods | 10.1038/s41592-019-0619-0 | ❌ Missing |
| scVI | Lopez et al. (2018) Nature Methods | 10.1038/s41592-018-0229-2 | ❌ Missing |
| Leiden clustering | Traag et al. (2019) Scientific Reports | 10.1038/s41598-019-41695-z | ❌ Missing |

---

## Recommended Retrieval Priority

### Batch 1: Foundational Tools (15 papers)
```
10.1186/s13059-017-1382-0  # Scanpy
10.1016/j.cell.2019.05.031  # Seurat v3
10.1038/s41587-021-01139-4  # Cell2location
10.1038/s41592-021-01358-2  # Squidpy
10.1038/s41592-024-02212-x  # SpatialData
10.1038/s41592-020-01018-x  # Cellpose
10.1038/s41587-021-01206-w  # scvi-tools
10.1038/s41592-021-01264-7  # Tangram
```

### Batch 2: H&E Prediction (8 papers)
```
10.1038/s41467-020-17678-4  # HE2RNA
10.1038/s41467-020-20249-2  # ST-Net
10.1038/s41592-024-02284-9  # BLEEP
10.1038/s41588-024-01773-9  # iStar
10.1038/s41698-022-00315-2  # Hist2ST
```

### Batch 3: Pathology Foundation Models (8 papers)
```
10.1038/s41591-024-02857-3  # UNI
10.1038/s41591-024-02856-4  # Virchow
10.1038/s41586-024-07441-w  # GigaPath
10.1038/s41591-023-02504-3  # PLIP
10.1038/s41591-024-02886-6  # CONCH
```

### Batch 4: Spatial Platforms (10 papers)
```
10.1126/science.aaf2403     # Original spatial (Stahl)
10.1126/science.aaw1219     # Slide-seq
10.1126/science.aaa6090     # MERFISH
10.1038/s41586-019-1049-y   # seqFISH+
10.1038/s41587-022-01483-z  # CosMx
```

### Batch 5: DL Foundations (9 papers)
```
10.48550/arXiv.1706.03762   # Attention Is All You Need
10.1007/978-3-319-24574-4_28  # U-Net
10.48550/arXiv.2010.11929   # ViT
10.48550/arXiv.2104.14294   # DINO
10.48550/arXiv.2304.07193   # DINOv2
```

---

## Retrieval Commands

### Use CORE API (open access)
```bash
cd /home/user/polymath-v4
python scripts/discover_papers.py "scanpy single cell analysis" --year-min 2018 --limit 10 --auto-ingest
python scripts/discover_papers.py "spatial transcriptomics deconvolution" --year-min 2020 --auto-ingest
```

### Manual retrieval needed for
- Nature Methods (many paywalled)
- Cell (paywalled)
- Science (paywalled)

Check Unpaywall/Sci-Hub for open access versions.

---

## Repository Coverage (GOOD ✅)

We already have these key repos indexed:
- scverse/scanpy (2,328 stars)
- satijalab/seurat (2,614 stars)
- mouseland/cellpose (2,032 stars)
- stardist/stardist (1,160 stars)
- scverse/squidpy (547 stars)
- bayraktarlab/cell2location (412 stars)
- broadinstitute/tangram (348 stars)
- scverse/spatialdata (339 stars)
- yoseflab/scvi-tools (1,538 stars)

**Consider adding:**
- mahmoodlab/HIPT
- mahmoodlab/UNI
- microsoft/GigaPath (if released)
- owkin/HistoSSL-related repos
