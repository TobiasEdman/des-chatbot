# DES Chatbot — Teknisk Specifikation

**Version:** 1.0.0
**Datum:** 2026-03-26
**Status:** Deployed on ICE Connect EKC

---

## 1. Arkitekturöversikt

```
 digitalearth.se (WordPress)
       |
       | [des-chatbot-widget.php]
       | JavaScript EventSource (SSE)
       v
 chat.digitalearth.se (NGINX Ingress + TLS)
       |
       v
 +-----------+       +----------+       +------------------+
 | RAG API   | ----> | Qdrant   |       | vLLM             |
 | FastAPI   |       | v1.12.1  |       | Mistral-7B-AWQ   |
 | 2 replicas|       | 10Gi PVC |       | 2080ti (11GB)    |
 | 8080/TCP  |       | 6333/TCP |       | 8000/TCP         |
 +-----------+       +----------+       +------------------+
       |                  ^
       |  cosine search   |
       +------------------+
       |
       | OpenAI-compat /chat/completions (streaming)
       +-------------------------------------------->  vLLM
```

---

## 2. Komponenter

### 2.1 LLM Inference — vLLM + Mistral-7B

| Parameter | Värde |
|---|---|
| **Modell** | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` |
| **Kvantisering** | AWQ (4-bit) |
| **vLLM-version** | `v0.6.6.post1` |
| **GPU** | NVIDIA GTX 2080 Ti (11 GB VRAM) |
| **Max context** | 4096 tokens |
| **GPU memory utilization** | 0.90 |
| **dtype** | float16 |
| **Execution mode** | `--enforce-eager` (krävs för CUDA CC 7.5) |
| **Max output tokens** | 1024 |
| **Temperature** | 0.3 |
| **API** | OpenAI-kompatibelt (`/v1/chat/completions`) |
| **K8s Deployment** | 1 replika, `nodeSelector: accelerator: nvidia-gtx-2080ti` |
| **Resurser** | 4 CPU, 16Gi RAM, 1 GPU |
| **Health probes** | Readiness: 300s initial delay, Liveness: 360s initial delay |

**Kända begränsningar:**
- `torch.compile` crashar på CUDA compute capability 7.5 — löst med `--enforce-eager`
- K8s Service heter `llm-inference` (inte `vllm`) för att undvika `VLLM_PORT` env-kollision
- Mistral v0.2 saknar native system-roll — system prompt bäddas in i user-meddelande

### 2.2 Vektordatabas — Qdrant

| Parameter | Värde |
|---|---|
| **Version** | `v1.12.1` |
| **Collection** | `des_knowledge` |
| **Distansmått** | Cosine similarity |
| **Vektordimension** | 384 |
| **Lagring** | 10Gi PVC (rook-ceph-rbd) |
| **K8s StatefulSet** | 1 replika |
| **Resurser** | 2 CPU, 4Gi RAM |
| **Portar** | HTTP: 6333, gRPC: 6334 |
| **Indexerade chunks** | 168 (nuvarande) |

### 2.3 Embedding-modell

| Parameter | Värde |
|---|---|
| **Modell** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimension** | 384 |
| **Inladdning** | Pre-downloaded i Docker-image vid build |
| **Runtime** | Körs i RAG API-podden (CPU) |

### 2.4 RAG API — FastAPI Backend

| Parameter | Värde |
|---|---|
| **Framework** | FastAPI 0.115.6 |
| **Runtime** | Uvicorn (1 worker per replika) |
| **Python** | 3.11-slim |
| **K8s Deployment** | 2 replikor |
| **Resurser** | 4 CPU, 8Gi RAM per pod |
| **Port** | 8080/TCP |
| **Streaming** | Server-Sent Events (SSE) |

**Endpoints:**

| Metod | Path | Beskrivning |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Chat med SSE-streaming |

**Request-format (`POST /api/chat`):**
```json
{
  "message": "Vad är Digital Earth Sweden?",
  "session_id": "optional-uuid"
}
```

**Response:** SSE-stream med events:
```
event: metadata
data: {"session_id": "uuid"}

data: {"text": "Digital Earth Sweden..."}
data: {"text": "\n\n---\n**Källor:**\n- [DES](https://...)"}

event: done
data: {"sources": [{"url": "https://...", "title": "...", "score": 0.72}]}
```

---

## 3. RAG-pipeline

### 3.1 Retrieval

1. User query → embedding via `all-MiniLM-L6-v2`
2. Cosine similarity-sökning i Qdrant (`top_k=5`)
3. **Score-filtrering**: chunks med `score < 0.30` filtreras bort
4. Om inga chunks passerar filtret → direkt svar utan LLM-anrop:
   *"Jag har ingen information om detta i tillgänglig kontext."*

### 3.2 System Prompt (strict, no-hallucination)

```
Du är DES Chatbot, en domänexpert inom Digital Earth Sweden (DES).
Du svarar på svenska om inte användaren skriver på ett annat språk.

Du svarar enbart på frågor som rör Digital Earth Swedens datamängder,
tjänster, openEO API, STAC, tutorials, guider, satellitdata,
fjärranalys och jordobservation — baserat på tillhandahållen kontext.

REGLER:
- Ge faktabaserade, kortfattade och seriösa svar.
- Ställ INGA uppföljningsfrågor och be INTE om förtydliganden.
- Förklara eller diskutera INTE saker som inte explicit efterfrågats.
- Spekulera eller gissa INTE om något saknas i kontexten.
- Inkludera INTE information från externa eller ej angivna källor.
- Om kontexten inte innehåller tillräcklig information, svara:
  "Jag har ingen information om detta i tillgänglig kontext."
- Ange alltid källa till svaret där det är möjligt.
- Avsluta svaret efter att frågan är besvarad.
```

### 3.3 Prompt Construction

```
[System prompt + regler (ovan)]

--- KONTEXT ---
Använd ENBART följande information för att svara:

[1] Källa: https://...
    Titel: ...
    Relevans: 0.72
    {chunk_text_1}

---

[2] Källa: https://...
    Titel: ...
    Relevans: 0.65
    {chunk_text_2}

--- SLUT KONTEXT ---

Användarens fråga: {query}
```

### 3.4 Svarsformat

Svar levereras med automatiska källhänvisningar:
```
[Svar baserat på kontext]

---
**Källor:**
- [Titel 1](https://source1.se)
- [Titel 2](https://source2.se)
```

**Stop-sekvenser:** `\n\nFråga:`, `\n\nUser:`, `\n\nAnvändare:` — förhindrar modellen från att generera egna följdfrågor.

### 3.5 Generation

- vLLM OpenAI-kompatibelt API med streaming
- Async httpx-klient med 120s timeout
- Temperature: **0.1** (minimal kreativitet, maximal precision)
- Svar streamas token-för-token tillbaka via SSE
- Källhänvisningar appendas automatiskt efter LLM-svaret
- Om inga chunks passerar score-filter → **inget LLM-anrop** (sparar GPU)

---

## 4. Content Indexering

### 4.1 Indexer (Python CLI)

```bash
python indexer.py wordpress    # Crawla digitalearth.se via WP REST API
python indexer.py stac         # Indexera DES STAC-katalog
python indexer.py markdown <dir>  # Lokala markdown-filer
python indexer.py all          # Alla ovanstående
```

### 4.2 Chunking

| Parameter | Värde |
|---|---|
| **Chunk size** | 512 ord |
| **Overlap** | 64 ord |
| **Metod** | Whitespace-baserad tokenisering |
| **ID-generering** | Deterministisk MD5 (source + index + prefix) |
| **Batch upsert** | 100 punkter per batch |

### 4.3 Kunskapskällor (3 nivåer)

**Tier 1 — Auto-indexerade (crawlas automatiskt):**
- `digitalearth.se` — WordPress sidor och inlägg
- `explorer.digitalearth.se/stac` — STAC-katalogen (dagligen)
- DES GitHub-repon (community, openeo-processes)
- ImintEngine (README + analyzer-docs, exkl. CLAUDE.md och training/)

**Tier 2 — Kurerade (51 källor):**
- ESA Sentinel teknisk dokumentation (S1, S2, S3)
- Copernicus Data Space, HR-VPP
- openEO API-specifikation
- Svenska myndigheter: Rymdstyrelsen, Naturvårdsverket (NMD), Jordbruksverket (LPIS), SMHI
- Vetenskapliga artiklar (LULC, crop mapping, InSAR, vessel detection, VedgeSat)
- Foundation models: Prithvi (IBM/NASA), Clay
- Svensk universitetsforskning: Lund, Chalmers
- DES GitHub-repon (ML, AI pipelines)

**Tier 3 — Blockerade:**
- Sociala medier, Reddit, Medium, StackOverflow, ChatGPT, Claude

---

## 5. WordPress-plugin

### 5.1 Plugin-info

| Parameter | Värde |
|---|---|
| **Namn** | DES Chatbot |
| **Fil** | `des-chatbot-widget.php` |
| **Version** | 1.0.0 |
| **Licens** | GPL-2.0+ |
| **Shortcode** | `[des_chatbot]` |
| **Auto-inject** | Via `wp_footer` hook |

### 5.2 Funktionalitet

- **Floating chat-bubbla** (fixed position, nedre höger)
- **Responsiv** (full-screen på mobil < 480px)
- **SSE-streaming** via EventSource (native browser API)
- **Markdown-rendering**: code blocks, bold, italic, länkar, listor
- **Session management**: `sessionStorage`-baserat session_id
- **Escape-tangent** stänger panelen
- **Admin-inställningar**: Konfigurerbar API URL under `Settings → DES Chatbot`
- **Tillgänglighet**: ARIA-labels, keyboard navigation, `aria-live="polite"`
- **DES-branding**: `--des-blue: #1a4a6e`, `--des-green: #2d8c5a`

---

## 6. Kubernetes-infrastruktur

### 6.1 Namespace & Deploy

```bash
# Full deploy med kustomize
kubectl apply -k des-chatbot/k8s/

# Enskilda resurser
kubectl apply -f des-chatbot/k8s/namespace.yaml
kubectl apply -f des-chatbot/k8s/secrets.yaml
kubectl apply -f des-chatbot/k8s/qdrant-statefulset.yaml
kubectl apply -f des-chatbot/k8s/vllm-deployment.yaml
kubectl apply -f des-chatbot/k8s/rag-api-deployment.yaml
kubectl apply -f des-chatbot/k8s/ingress.yaml
```

### 6.2 Resurskrav (totalt)

| Komponent | CPU | RAM | GPU | Storage |
|---|---|---|---|---|
| vLLM | 4 | 16Gi | 1x 2080ti | — |
| RAG API (x2) | 8 | 16Gi | — | — |
| Qdrant | 2 | 4Gi | — | 10Gi PVC |
| **Totalt** | **14** | **36Gi** | **1 GPU** | **10Gi** |

### 6.3 Ingress

| Parameter | Värde |
|---|---|
| **Domän** | `chat.digitalearth.se` |
| **TLS** | Let's Encrypt via cert-manager |
| **Ingress controller** | NGINX |
| **Rate limiting** | 10 req/s, burst 20, max 5 connections |
| **Proxy timeout** | 120s (read + send) |
| **Max body** | 10MB |

### 6.4 Säkerhet

- **CORS**: Begränsat till `digitalearth.se` och `www.digitalearth.se`
- **Rate limiting**: Dubbelt — NGINX Ingress (10 req/s) + FastAPI (10 req/min per IP)
- **Secrets**: HuggingFace token i K8s Secret (base64)
- **Inga lösenord i images**: Token injiceras via env vars
- **TLS**: End-to-end HTTPS via Let's Encrypt

---

## 7. Python Dependencies

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
httpx==0.28.1
qdrant-client==1.13.3
sentence-transformers==3.3.1
torch==2.5.1
```

---

## 8. Konfiguration (Environment Variables)

| Variabel | Default | Beskrivning |
|---|---|---|
| `VLLM_URL` | `http://vllm:8000/v1` | vLLM API endpoint |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | HF modell-ID |
| `LLM_MAX_TOKENS` | `1024` | Max output tokens |
| `LLM_TEMPERATURE` | `0.3` | Genererings-temperatur |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint |
| `COLLECTION_NAME` | `des_knowledge` | Qdrant collection-namn |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding-modell |
| `CHUNK_SIZE` | `512` | Ord per chunk |
| `CHUNK_OVERLAP` | `64` | Överlappande ord |
| `RETRIEVAL_TOP_K` | `5` | Antal chunks från Qdrant |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.30` | Min cosine score för att behålla chunk |
| `RATE_LIMIT_PER_MINUTE` | `10` | Max requests/min per IP |
| `MAX_HISTORY_MESSAGES` | `5` | Konversationsrundor i minne |
| `LOG_LEVEL` | `INFO` | Python log level |

---

## 9. Docker Build

```dockerfile
FROM python:3.11-slim
# Pre-downloads embedding model (384-dim, ~90MB)
# Exposes port 8080
# CMD: uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
```

```bash
docker build -t des-chatbot-api:latest -f Dockerfile .
```

---

## 10. Kända problem och förbättringar

### Lösta problem (v1.0 → v1.1)
1. ~~**Yviga svar**~~ → Strikt system prompt med explicita REGLER-block, temperature 0.1, stop-sekvenser
2. ~~**Inga source citations**~~ → Automatiska källhänvisningar efter varje svar
3. ~~**Irrelevant kontext**~~ → Score-filtrering (threshold 0.30) + no-context fallback utan LLM-anrop
4. ~~**GET-baserat chat-API**~~ → WordPress-widget använder nu `fetch` POST med SSE-parsing

### Kvarstående problem
1. **In-memory session store** — Konversationshistorik förloras vid pod-restart
2. **Mistral v0.2 system-roll** — Saknar native system role, bäddas in i user-meddelande

### Planerade förbättringar
- [ ] Redis-backed session store
- [ ] GitHub Actions CI/CD pipeline
- [ ] Scheduled re-indexering (cron job i K8s)
- [ ] Monitoring med Prometheus metrics
- [ ] Uppgradera till större modell om H100 blir tillgänglig
- [ ] GitHub-repo indexering (README/docs-crawling)
- [ ] Feedback-knapp (tumme upp/ner) i chat-widgeten
