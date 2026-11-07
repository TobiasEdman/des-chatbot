# des-chatbot

User-facing Digital Earth Sweden chatbot. A FastAPI service retrieves curated
DES content from Qdrant, sends grounded context to a vLLM-served language model,
and streams answers to the WordPress widget over Server-Sent Events.

This service is operationally separate from `omni-rag`: `des-chatbot` owns the
public API and its production index, while `omni-rag` is local developer
retrieval infrastructure.

## Architecture

```text
WordPress widget
      │ POST /api/chat (SSE)
      ▼
FastAPI ──> Qdrant ──> vLLM
   │           │
   └──────── Neo4j graph retrieval
```

The active Kustomize bundle under `k8s/` deploys the API, Qdrant, vLLM,
ingress, and certificate configuration in `prithvi-training-default`. A Neo4j
StatefulSet manifest is present but is not currently included by
`k8s/kustomization.yaml`.

## API

- `GET /api/health` — process health.
- `POST /api/chat` — accepts `message` and optional `session_id`; returns an SSE
  stream containing metadata, text chunks, and a final source list.

Conversation history and rate-limit state are currently in process memory.
They are not shared between workers and do not survive a restart.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
NEO4J_PASSWORD='<local-password>' uvicorn main:app --reload --port 8080
```

The default service URLs target the Kubernetes service names. Override
`QDRANT_URL`, `VLLM_URL`, and related settings when running dependencies
locally. Configuration and defaults are defined in `backend/config.py`.

## Index content

Curated source definitions live in `backend/curated_sources.yaml`. The indexer
and raw seed material are under `backend/indexer.py` and `data/raw/`. Confirm
source licensing, sensitivity, and freshness before adding or rebuilding data.

The production Qdrant collection currently uses a 384-dimensional multilingual
sentence-transformer. `des-contracts` declares a different canonical embedding
configuration; migration requires a deliberate full reindex and must not be
performed as an in-place model switch.

## Verify

```bash
python -m compileall -q backend
docker build -t des-chatbot:local .
kubectl kustomize k8s/ >/dev/null
```

CI verifies that Neo4j credentials fail fast when missing and that the graph
module imports when a sentinel credential is supplied. It also runs backend
tests when `backend/tests/` exists; no backend test directory is currently
present.

## Deployment and operations

- Technical design: [`docs/technical-specification.md`](docs/technical-specification.md)
- ICE networking: [`docs/ice-networking.md`](docs/ice-networking.md)
- WordPress installation: [`docs/wordpress-installation.md`](docs/wordpress-installation.md)
- Widget: [`wordpress/des-chatbot-widget.php`](wordpress/des-chatbot-widget.php)

Never commit real values to `k8s/secrets.yaml`, `.env`, documentation examples,
or generated distribution archives. Rotate credentials through the cluster's
secret-management procedure.
