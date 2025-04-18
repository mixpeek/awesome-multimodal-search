# Awesome-Multimodal-Search

A curated collection of 🔍 libraries, ☁️ platforms, 📖 research, 📊 benchmarks, and 📚 tutorials focused on **Multimodal Search** — enabling semantic retrieval across images, video, audio, and documents.

---

## 🔍 Libraries & Frameworks

| Name | Description | Links |
|------|-------------|--------|
| **Jina AI** | Flow-based neural search framework for text, image, video, and audio. | [GitHub](https://github.com/jina-ai/jina) · [Website](https://jina.ai) |
| **Weaviate** | Vector DB with modules for image, text, and audio embeddings (e.g. CLIP, ImageBind). | [GitHub](https://github.com/weaviate/weaviate) · [Website](https://weaviate.io) |
| **Towhee** | Multimodal data pipelines with 100+ pretrained models. | [GitHub](https://github.com/towhee-io/towhee) · [Website](https://towhee.io) |
| **CLIP Retrieval** | Lightweight toolkit to search CLIP-embedded LAION datasets. | [GitHub](https://github.com/rom1504/clip-retrieval) · [Demo](https://clip-retrieval.org) |

---

## ☁️ Cloud Services & APIs

| Name | Modalities | Links | Notes |
|------|------------|-------|-------|
| **OpenAI API** | Text, image (GPT-4V), audio (Whisper) | [Docs](https://platform.openai.com/docs) | Supports RAG + embeddings |
| **Vertex AI (Google)** | Image + Text | [Docs](https://cloud.google.com/vertex-ai/docs/matching-engine/overview) | CoCa model embeddings |
| **AWS Rekognition + Kendra + Transcribe** | Image, text, audio | [Rekognition](https://aws.amazon.com/rekognition) · [Kendra](https://aws.amazon.com/kendra) | Modular pipeline for multimodal search |

---

## 📖 Landmark Papers

| Title | Modality | Venue | Links |
|-------|----------|--------|--------|
| **CLIP** | Image–Text | ICML 2021 | [Paper](https://arxiv.org/abs/2103.00020) · [GitHub](https://github.com/openai/CLIP) |
| **ImageBind** | All (6 modalities) | ICML 2023 | [Paper](https://arxiv.org/abs/2305.05665) · [GitHub](https://github.com/facebookresearch/ImageBind) |
| **CLAP** | Audio–Text | NeurIPS 2022 | [Paper](https://arxiv.org/abs/2301.12503) · [GitHub](https://github.com/LAION-AI/CLAP) |

---

## 📊 Benchmarks & Leaderboards

| Benchmark | Modality | Metric | Example |
|-----------|----------|--------|---------|
| **MS COCO** | Image–Text | R@1, R@5, R@10 | BLIP-2 > 80% R@1 |
| **MSR-VTT** | Video–Text | R@1, R@5 | Marengo > 60% R@1 |
| **Clotho, AudioCaps** | Audio–Text | mAP@10, R@10 | CLAP ~0.21 mAP |
| **Wiki-SS** | Document Screenshots | Top-1 Accuracy | DSE 49% top-1 |

---

## 📚 Tutorials & Demos

| Title | Modality | Links |
|-------|----------|-------|
| **ImageBind + Deep Lake** | Unified search | [Tutorial](https://docs.deeplake.ai/en/latest/deeplake/imagebind.html) |
| **Pinecone + CLIP** | Text–Image | [Blog](https://www.pinecone.io/learn/multimodal-search/) |
| **Jina Hello Multimodal** | Text + Image | [Code](https://github.com/jina-ai/jina) |
| **RAG + CLIP + OpenAI** | Multimodal RAG | [Colab](https://platform.openai.com/examples) |

---

📬 **Contributions welcome!** PRs and issues encouraged.
