# Replicating DeepLX: Free DeepL Translation in Rust

This document explains how [DeepLX](https://github.com/OwO-Network/DeepLX) (Go) achieves token-free DeepL translation and how to replicate the approach in Rust.

## Core Concept

DeepLX bypasses DeepL's paid API by impersonating the **official DeepL Chrome Extension** (ID: `cofdbpoegempjloogbagkncekinflcnj`). The extension uses an internal "oneshot" endpoint that accepts anonymous requests with `Authorization: None` — a completely separate rate-limit pool from the `www2.deepl.com` web translator.

## The OneShot Endpoint

```
POST https://oneshot-free.www.deepl.com/v1/translate    (free / anonymous)
POST https://oneshot-pro.www.deepl.com/v1/translate     (pro, with OAuth token)
```

This is reverse-engineered from the extension's `background.js`. The old `LMT_handle_texts` approach on `www2.deepl.com` now 429s anonymous traffic almost immediately; the `oneshot` endpoint lives on a different CDN/pool.

## TLS Fingerprinting & HTTP Impersonation

The most critical part: **your HTTP client must look exactly like Chrome 120's `fetch()`**.

### Why this matters
DeepL's WAF checks:
- TLS ClientHello cipher suites, extensions, and JA3/JA4 fingerprint
- HTTP/2 SETTINGS frame order
- Pseudo-header order
- `User-Agent` / `sec-ch-ua` headers consistency with TLS version
- Presence/absence of browser-specific headers

A single mismatch is a cheap signal for blocking. In DeepLX's Go implementation, the `req` library's `ImpersonateChrome()` handles the TLS and HTTP/2 fingerprinting.

### Rust equivalents

| Feature | Go (DeepLX) | Rust |
|---------|------------|------|
| TLS fingerprint | `req.C().ImpersonateChrome()` | [`rquest`](https://crates.io/crates/rquest) (fork of reqwest with TLS impersonation) or [`tls_client`](https://crates.io/crates/tls_client) |
| HTTP/2 fingerprint | handled by impersonation | `rquest` supports `rquest::Client::builder().impersonate(Impersonate::Chrome120)` |
| Cookie jar | `cookiejar.New(nil)` | Built into `rquest`/`reqwest` |
| Connection pooling | `sync.Map` per proxy | Built into `rquest`/`reqwest` |
| Compression | manual gzip/deflate/br | `rquest` handles it (or disable auto-decompress + use crates) |

**Recommended**: Use [`rquest`](https://crates.io/crates/rquest) which is a reqwest fork with TLS/JA3 impersonation built in.

```toml
[dependencies]
rquest = "0.28"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4"] }
tokio = { version = "1", features = ["full"] }
```

## Step-by-Step Implementation

### 1. Create a Chrome-impersonated HTTP client

```rust
use std::sync::OnceLock;
use rquest::{Client, Impersonate};
use std::time::Duration;

const ONESHOT_TIMEOUT: Duration = Duration::from_secs(20);
const WARMUP_TIMEOUT: Duration = Duration::from_secs(5);

fn get_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .impersonate(Impersonate::Chrome120)
            .cookie_store(true)
            .timeout(ONESHOT_TIMEOUT)
            .no_brotli()  // Handle decompression manually if needed
            .no_gzip()
            .no_deflate()
            .build()
            .expect("Failed to build HTTP client")
    })
}
```

### 2. Warm cookies (GET www.deepl.com once)

The extension's `fetch()` inherits whatever cookies the browser has accumulated on `.deepl.com`. A cold visit to `www.deepl.com` sets `userCountry=<iso2>` and `verifiedBot=false`. Subsequent `oneshot` POSTs to the same eTLD+1 automatically carry those cookies.

```rust
use std::sync::Once;

static COOKIE_WARMER: Once = Once::new();

async fn warm_cookies(client: &Client) {
    COOKIE_WARMER.call_once(|| {
        // Fire-and-forget in background; cookies are best-effort
        tokio::spawn(async move {
            let _ = client
                .get("https://www.deepl.com/translator")
                .timeout(WARMUP_TIMEOUT)
                .send()
                .await;
        });
    });
}
```

### 3. Generate a stable instance ID (UUID v4)

The Chrome extension persists a UUID in `chrome.storage` on install. DeepLX generates one per process lifetime.

```rust
use uuid::Uuid;

fn instance_id() -> String {
    Uuid::new_v4().to_string()
}
```

### 4. Build the request body

```rust
use serde::Serialize;

#[derive(Serialize)]
struct AppInformation {
    os: String,
    os_version: String,
    app_version: String,
    app_build: String,
    instance_id: String,
}

#[derive(Serialize)]
struct OneshotRequest {
    text: Vec<String>,
    target_lang: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    source_lang: String,
    usage_type: String,
    app_information: AppInformation,
}

fn build_request(text: &str, source_lang: &str, target_lang: &str) -> OneshotRequest {
    OneshotRequest {
        text: vec![text.to_string()],
        target_lang: target_lang.to_string(),
        source_lang: source_lang.to_string(), // empty = autodetect, omitempty drops it
        usage_type: "Translate".to_string(),
        app_information: AppInformation {
            os: "brex_macOS".to_string(),
            os_version: "brex_chrome_120.0.0.0".to_string(),
            app_version: "1.86.0".to_string(),
            app_build: "chrome_web_store".to_string(),
            instance_id: instance_id(),
        },
    }
}
```

### 5. Send the request with exact Chrome Extension headers

```rust
const CHROME_EXTENSION_ID: &str = "cofdbpoegempjloogbagkncekinflcnj";
const ONESHOT_FREE_ENDPOINT: &str = "https://oneshot-free.www.deepl.com/v1/translate";
const MAX_FREE_TEXT_LENGTH: usize = 1500; // hard cap enforced by oneshot

async fn translate(
    client: &Client,
    text: &str,
    source_lang: &str,
    target_lang: &str,
    dl_session: Option<&str>, // Some(token) = Pro mode
) -> Result<TranslationResult, String> {
    if text.is_empty() {
        return Err("No text to translate".into());
    }
    if text.chars().count() > MAX_FREE_TEXT_LENGTH {
        return Err(format!(
            "Text exceeds maximum length: {} characters (limit is {})",
            text.chars().count(),
            MAX_FREE_TEXT_LENGTH
        ));
    }

    let endpoint = if dl_session.is_some() {
        "https://oneshot-pro.www.deepl.com/v1/translate"
    } else {
        ONESHOT_FREE_ENDPOINT
    };

    let auth_value = match dl_session {
        Some(token) => format!("Bearer {}", token),
        None => "None".to_string(), // literal string "None" for anonymous
    };

    let body = build_request(text, source_lang, target_lang);
    let body_bytes = serde_json::to_vec(&body).unwrap();

    let response = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .header("Accept", "*/*")
        .header("Authorization", &auth_value)
        .header("Origin", format!("chrome-extension://{}", CHROME_EXTENSION_ID))
        .header("Sec-Fetch-Site", "cross-site")
        .header("Sec-Fetch-Mode", "cors")
        .header("Sec-Fetch-Dest", "empty")
        .header("Accept-Encoding", "gzip, deflate, br")
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    // Parse response...
    // (handle gzip/deflate/br decompression yourself if you disabled auto-decompress)
}
```

### 6. Header cleanup

DeepLX explicitly **removes** these headers that `req`'s impersonation adds but a `fetch()` never emits:
- `Pragma`
- `Cache-Control`
- `Upgrade-Insecure-Requests`
- `Sec-Fetch-User`

With `rquest`, you may need similar cleanup depending on what the impersonation preset includes.

### 7. Response structure

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OneshotTranslation {
    text: String,
    detected_source_language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OneshotResponse {
    translations: Vec<OneshotTranslation>,
}

#[derive(Debug, Serialize)]
struct TranslationResult {
    code: u16,
    id: i64,
    message: Option<String>,
    data: String,
    alternatives: Vec<String>,
    source_lang: String,
    target_lang: String,
    method: String,
}
```

## Language Codes

The oneshot endpoint uses lowercase BCP-47-ish codes. DeepLX maintains two maps:

### Target languages (what the API accepts as `target_lang`)
```
AR→ar  BG→bg  CS→cs  DA→da  DE→de  EL→el  EN-GB→en-GB  EN-US→en-US
ES→es  ES-419→es-419  ET→et  FI→fi  FR→fr  HE→he  HU→hu  ID→id
IT→it  JA→ja  KO→ko  LT→lt  LV→lv  NB→nb  NL→nl  PL→pl
PT-BR→pt-BR  PT-PT→pt-PT  RO→ro  RU→ru  SK→sk  SL→sl  SV→sv
TR→tr  UK→uk  VI→vi  ZH→zh-Hans  ZH-HANS→zh-Hans  ZH-HANT→zh-Hant
```

Convenience aliases: `EN` → `en-US`, `PT` → `pt-BR`

### Source languages (superset of target)
Same as above plus: `EN` → `en`, `PT` → `pt`

## Key Constraints

| Constraint | Value | Source |
|-----------|-------|--------|
| Max text length (free) | 1,500 chars | `G.notLoggedIn = 1500` in extension's `background.js` |
| Translate timeout | 20s | Observed field behavior |
| Cookie warmup timeout | 5s | Best-effort |
| Rate limit (free) | Unknown, but 429s quickly | Separate pool from web translator |

## Connection Pooling (Performance)

DeepLX caches one HTTP client per proxy URL in a `sync.Map`, reusing TCP/TLS/HTTP2 connections. Without this, every request triggers a fresh TLS handshake (~200-400ms). With pooling and session tickets, subsequent requests skip the handshake entirely.

In Rust, `rquest`/`reqwest` handles this automatically via its internal connection pool.

## Complete Rust Example

A minimal working example using `rquest`:

```rust
use rquest::{Client, Impersonate};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

const EXTENSION_ID: &str = "cofdbpoegempjloogbagkncekinflcnj";
const FREE_ENDPOINT: &str = "https://oneshot-free.www.deepl.com/v1/translate";

#[derive(Serialize)]
struct AppInfo {
    os: String,
    os_version: String,
    app_version: String,
    app_build: String,
    instance_id: String,
}

#[derive(Serialize)]
struct Req {
    text: Vec<String>,
    target_lang: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    source_lang: String,
    usage_type: String,
    app_information: AppInfo,
}

#[derive(Deserialize)]
struct TransItem {
    text: String,
    detected_source_language: Option<String>,
}

#[derive(Deserialize)]
struct Resp {
    translations: Vec<TransItem>,
}

fn build(text: &str, source: &str, target: &str) -> Req {
    Req {
        text: vec![text.to_string()],
        target_lang: target.to_string(),
        source_lang: source.to_string(),
        usage_type: "Translate".to_string(),
        app_information: AppInfo {
            os: "brex_macOS".into(),
            os_version: "brex_chrome_120.0.0.0".into(),
            app_version: "1.86.0".into(),
            app_build: "chrome_web_store".into(),
            instance_id: Uuid::new_v4().to_string(),
        },
    }
}

async fn translate(text: &str, target_lang: &str) -> Result<String, String> {
    let client = Client::builder()
        .impersonate(Impersonate::Chrome120)
        .cookie_store(true)
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;

    // Warm cookies (best-effort, fire-and-forget)
    let _ = client
        .get("https://www.deepl.com/translator")
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    let body = serde_json::to_vec(&build(text, "", target_lang)).unwrap();

    let resp = client
        .post(FREE_ENDPOINT)
        .header("Content-Type", "application/json")
        .header("Accept", "*/*")
        .header("Authorization", "None")
        .header("Origin", format!("chrome-extension://{}", EXTENSION_ID))
        .header("Sec-Fetch-Site", "cross-site")
        .header("Sec-Fetch-Mode", "cors")
        .header("Sec-Fetch-Dest", "empty")
        .body(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let parsed: Resp = resp.json().await.map_err(|e| e.to_string())?;
    parsed
        .translations
        .first()
        .map(|t| t.text.clone())
        .ok_or_else(|| "No translation returned".to_string())
}

#[tokio::main]
async fn main() {
    match translate("Hello, world!", "DE").await {
        Ok(t) => println!("Translated: {}", t),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Summary Checklist

1. **Use Chrome 120 TLS fingerprint** via `rquest`/`rustls` impersonation
2. **Send `Authorization: None`** (literal string) for free access
3. **Set `Origin: chrome-extension://cofdbpoegempjloogbagkncekinflcnj`**
4. **Include `app_information`** with `brex_macOS` / `brex_chrome_120.0.0.0` fields
5. **Warm cookies** by visiting `www.deepl.com/translator` once
6. **Reuse the HTTP client** for connection pooling
7. **Stay under 1,500 characters** per translation request
8. **Throttle requests** — the free endpoint will 429 after excessive use
