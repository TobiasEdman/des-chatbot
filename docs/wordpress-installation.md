# DES Chatbot — WordPress Installation

## Förutsättningar

- WordPress 5.0+ (testat med 6.x)
- Backend-stacken deployad och nåbar via `chat.digitalearth.se`
- Admin-åtkomst till WordPress

---

## Installation

### Alternativ A: ZIP-upload (rekommenderat)

1. Logga in på WordPress admin (`digitalearth.se/wp-admin`)
2. Gå till **Plugins → Add New → Upload Plugin**
3. Välj `dist/des-chatbot.zip`
4. Klicka **Install Now**, sedan **Activate**

### Alternativ B: FTP/SSH

```bash
# Kopiera plugin-filen till WordPress plugins-katalog
scp wordpress/des-chatbot-widget.php user@server:/var/www/html/wp-content/plugins/des-chatbot/des-chatbot-widget.php
```

Aktivera sedan under **Plugins** i WordPress admin.

### Alternativ C: WP-CLI

```bash
wp plugin install dist/des-chatbot.zip --activate
```

---

## Konfiguration

1. Gå till **Settings → DES Chatbot** i WordPress admin
2. Ange API URL: `https://chat.digitalearth.se/api/chat`
3. Spara

Default-URL:en pekar redan på `https://chat.digitalearth.se/api/chat`.

---

## Verifiering

1. Öppna valfri sida på `digitalearth.se`
2. En blå chat-ikon ska visas i nedre högra hörnet
3. Klicka på ikonen — chatpanelen öppnas
4. Skriv en testfråga: *"Vad är Digital Earth Sweden?"*
5. Verifiera att svar streamas in i realtid

---

## Shortcode (valfritt)

Om du vill bädda in chatboten på en specifik sida istället för floating widget:

```
[des_chatbot]
```

> **OBS:** Pluginet injicerar widgeten automatiskt via `wp_footer`. Om du bara vill använda shortcoden, kommentera bort rad 57 i `des-chatbot-widget.php`:
> ```php
> // add_action('wp_footer', 'des_chatbot_render_widget');
> ```

---

## CORS-konfiguration

Backend-APIet tillåter bara requests från:
- `https://digitalearth.se`
- `https://www.digitalearth.se`

Om sajten körs på annan domän (t.ex. staging), lägg till env var i RAG API-deploymenten:

```yaml
env:
  - name: EXTRA_CORS_ORIGIN
    value: "https://staging.digitalearth.se"
```

---

## Felsökning

| Problem | Lösning |
|---|---|
| Chat-ikon syns inte | Kontrollera att pluginet är aktiverat under Plugins |
| "Something went wrong" | Kontrollera att RAG API körs: `curl https://chat.digitalearth.se/api/health` |
| CORS-fel i konsolen | Verifiera att domänen finns i `ALLOWED_ORIGINS` |
| Timeout vid svar | vLLM kan behöva starta om — kolla `kubectl logs -l app=vllm` |
| Tom chatbubbla | Kontrollera API URL under Settings → DES Chatbot |

---

## Avinstallation

1. **Plugins → Installed Plugins → DES Chatbot → Deactivate → Delete**
2. Alternativt: `wp plugin deactivate des-chatbot && wp plugin delete des-chatbot`
