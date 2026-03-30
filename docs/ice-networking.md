# ICE Connect — Nätverkskonfiguration

## Kubernetes Networking

### Service-typer

**ClusterIP** (default)
- Ger containern en intern IP inom klustret
- Exponerar porten **bara inom Kubernetes**
- Används för intern kommunikation (t.ex. RAG API → Qdrant, RAG API → vLLM)

**NodePort**
- Exponerar tjänsten **utanför Kubernetes** på alla noder
- Port **alltid i spannet 30000–32767**
- Nåbar via: `<nod-ip>:<nodeport>` eller `<domän>:<nodeport>`
- OBS: kräver portspecifikation i URL (t.ex. `chat.digitalearth.se:30135`)

### Ingress

- Länkar en **URL** (hostname) med en **Service**
- ICE pekar alla domäner `*.icedc.se` mot Kubernetes automatiskt
- Custom-domäner (t.ex. `chat.digitalearth.se`) kräver DNS-pekare mot:
  - **`213.21.96.181`** — ICE:s lastbalanserare för Kubernetes
- TLS/HTTPS-certifikat kan läggas till via cert-manager / Let's Encrypt

### DNS-konfiguration för chat.digitalearth.se

```
chat.digitalearth.se  →  A  213.21.96.181  (ICE K8s lastbalanserare)
```

**Alternativ utan Ingress (NodePort + reverse proxy):**
1. Skapa NodePort Service (port 30000-32767)
2. Konfigurera extern reverse proxy som tar emot `chat.digitalearth.se:443`
3. Proxyn routar till `213.21.96.181:<nodeport>`
4. HTTPS-certifikat på reverse proxyn

### Aktuell konfiguration

| Komponent | Service-typ | Port | Extern access |
|---|---|---|---|
| vLLM (llm-inference) | ClusterIP | 8000 | Nej (intern) |
| Qdrant | ClusterIP | 6333/6334 | Nej (intern) |
| RAG API | ClusterIP | 8080 | **Via Ingress → chat.digitalearth.se** |

### Deploy-steg för att gå live

1. ✅ DNS: Peka `chat.digitalearth.se` → `213.21.96.181`
2. Deploya RAG API (bygga Docker-image, pusha, skapa deployment)
3. Skapa Ingress: `kubectl apply -f k8s/ingress.yaml`
4. Verifiera TLS: `curl -v https://chat.digitalearth.se/api/health`
5. Installera WordPress-plugin: `dist/des-chatbot.zip`

### icedc.se-domäner

ICE pekar automatiskt alla `*.icedc.se`-domäner mot Kubernetes.
Kan användas som alternativ under utveckling:
```
chat.icedc.se → Ingress → rag-api:8080
```
