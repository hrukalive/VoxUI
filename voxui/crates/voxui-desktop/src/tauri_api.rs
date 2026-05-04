use serde::{de::DeserializeOwned, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = "invoke", catch)]
    async fn tauri_invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;
}

/// Invoke a Tauri command with typed arguments and return value.
pub async fn invoke<A: Serialize, R: DeserializeOwned>(cmd: &str, args: &A) -> Result<R, String> {
    let args_js = serde_wasm_bindgen::to_value(args).map_err(|e| e.to_string())?;
    let result = tauri_invoke(cmd, args_js)
        .await
        .map_err(|e| {
            e.as_string().unwrap_or_else(|| format!("{:?}", e))
        })?;
    serde_wasm_bindgen::from_value(result).map_err(|e| e.to_string())
}

/// Invoke a Tauri command that returns nothing.
pub async fn invoke_unit<A: Serialize>(cmd: &str, args: &A) -> Result<(), String> {
    let args_js = serde_wasm_bindgen::to_value(args).map_err(|e| e.to_string())?;
    tauri_invoke(cmd, args_js)
        .await
        .map_err(|e| e.as_string().unwrap_or_else(|| format!("{:?}", e)))?;
    Ok(())
}

/// Invoke a Tauri command with no arguments.
pub async fn invoke_no_args<R: DeserializeOwned>(cmd: &str) -> Result<R, String> {
    let args_js = serde_wasm_bindgen::to_value(&serde_json::json!({})).map_err(|e| e.to_string())?;
    let result = tauri_invoke(cmd, args_js)
        .await
        .map_err(|e| e.as_string().unwrap_or_else(|| format!("{:?}", e)))?;
    serde_wasm_bindgen::from_value(result).map_err(|e| e.to_string())
}

/// Extract the payload from a Tauri event object, accepting raw payloads for compatibility.
pub fn event_payload(value: JsValue) -> JsValue {
    match js_sys::Reflect::get(&value, &JsValue::from_str("payload")) {
        Ok(payload) if !payload.is_undefined() => payload,
        _ => value,
    }
}

// Event listening - we use raw JS via wasm_bindgen for Tauri event API
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"], js_name = "listen", catch)]
    pub async fn tauri_listen(event: &str, handler: &Closure<dyn FnMut(JsValue)>) -> Result<JsValue, JsValue>;
}
