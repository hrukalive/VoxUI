use std::collections::BTreeMap;
use std::io;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::Serialize;
use serde_json::Value;
use tungstenite::client::{uri_mode, IntoClientRequest};
use tungstenite::stream::{MaybeTlsStream, Mode};
use tungstenite::{client_tls_with_config, HandshakeError, Message, WebSocket};

pub const APP_ID: u64 = 1_651_388_990_835;
pub const HOST: &str = "https://live-open.biliapi.com";
pub const SIGN_URLS: [&str; 2] = [
    "https://soft.ceve-market.org/bopen/sign",
    "https://bopen.ceve-market.org/sign",
];
pub const CEVE_HEARTBEAT_URL: &str = "http://localhost.ceve-market.org:5218/heartbeat";
pub const HEARTBEAT_INTERVAL_SECS: u64 = 20;

const HEADER_LEN: u16 = 16;
const PROTOCOL_VERSION: u16 = 0;
const SEQUENCE_ID: u32 = 1;
const OP_HEARTBEAT: u32 = 2;
const OP_HEARTBEAT_REPLY: u32 = 3;
const OP_MESSAGE: u32 = 5;
const OP_AUTH: u32 = 7;
const OP_AUTH_REPLY: u32 = 8;
const READ_TIMEOUT: Duration = Duration::from_millis(500);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const STOP_CHECK_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerCommand {
    EndApp { game_id: String },
    Stop,
}

#[derive(Debug, Default)]
pub struct LiveWorkerState {
    game_id: Option<String>,
    end_sent: bool,
}

impl LiveWorkerState {
    pub fn for_test(game_id: Option<String>) -> Self {
        Self {
            game_id,
            end_sent: false,
        }
    }

    pub fn set_game_id(&mut self, game_id: String) {
        self.game_id = Some(game_id);
    }

    pub fn mark_disconnect_requested(&mut self) -> Option<WorkerCommand> {
        if self.end_sent {
            return None;
        }
        self.end_sent = true;
        self.game_id
            .clone()
            .map(|game_id| WorkerCommand::EndApp { game_id })
    }

    pub fn run_disconnect_cleanup<E>(
        &mut self,
        cleanup: impl FnOnce(&str) -> Result<(), E>,
    ) -> Result<(), E> {
        if let Some(WorkerCommand::EndApp { game_id }) = self.mark_disconnect_requested() {
            cleanup(&game_id)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct StartAppResponse {
    game_id: String,
    auth_body: String,
    websocket_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenBlivePacket {
    pub op: u32,
    pub body: Vec<u8>,
}

impl OpenBlivePacket {
    pub fn pack(&self) -> Vec<u8> {
        let packet_len = HEADER_LEN as usize + self.body.len();
        let packet_len = u32::try_from(packet_len).expect("OpenLive packet length must fit in u32");
        let mut bytes = Vec::with_capacity(packet_len as usize);

        bytes.extend_from_slice(&packet_len.to_be_bytes());
        bytes.extend_from_slice(&HEADER_LEN.to_be_bytes());
        bytes.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        bytes.extend_from_slice(&self.op.to_be_bytes());
        bytes.extend_from_slice(&SEQUENCE_ID.to_be_bytes());
        bytes.extend_from_slice(&self.body);

        bytes
    }
}

pub fn unpack_packet(bytes: &[u8]) -> anyhow::Result<OpenBlivePacket> {
    if bytes.len() < HEADER_LEN as usize {
        bail!(
            "OpenLive packet too short: {} bytes, expected at least {}",
            bytes.len(),
            HEADER_LEN
        );
    }

    let packet_len = u32::from_be_bytes(bytes[0..4].try_into()?) as usize;
    if packet_len != bytes.len() {
        bail!(
            "OpenLive packet length mismatch: header says {}, actual {}",
            packet_len,
            bytes.len()
        );
    }

    let header_len = u16::from_be_bytes(bytes[4..6].try_into()?);
    if header_len != HEADER_LEN {
        bail!(
            "unsupported OpenLive packet header length: {}, expected {}",
            header_len,
            HEADER_LEN
        );
    }

    let op = u32::from_be_bytes(bytes[8..12].try_into()?);
    let body = bytes[HEADER_LEN as usize..].to_vec();

    Ok(OpenBlivePacket { op, body })
}

pub fn unpack_packets(bytes: &[u8]) -> anyhow::Result<Vec<OpenBlivePacket>> {
    let mut packets = Vec::new();
    let mut offset = 0;

    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        if remaining < HEADER_LEN as usize {
            bail!(
                "OpenLive packet too short at offset {}: {} bytes, expected at least {}",
                offset,
                remaining,
                HEADER_LEN
            );
        }

        let packet_len = u32::from_be_bytes(bytes[offset..offset + 4].try_into()?) as usize;
        if packet_len < HEADER_LEN as usize {
            bail!(
                "OpenLive packet length too small at offset {}: header says {}, expected at least {}",
                offset,
                packet_len,
                HEADER_LEN
            );
        }
        let end = offset
            .checked_add(packet_len)
            .ok_or_else(|| anyhow!("OpenLive packet length overflow at offset {offset}"))?;
        if end > bytes.len() {
            bail!(
                "OpenLive packet length mismatch at offset {}: header says {}, remaining {}",
                offset,
                packet_len,
                remaining
            );
        }

        packets.push(unpack_packet(&bytes[offset..end])?);
        offset = end;
    }

    Ok(packets)
}

pub fn compact_json_body(value: &Value) -> anyhow::Result<String> {
    serde_json::to_string(&SortedJsonValue::from(value))
        .map_err(|error| anyhow!("failed to serialize compact OpenLive JSON body: {error}"))
}

pub fn run_openblive_worker(
    identity_code: String,
    enable_ceve_server_heartbeat: bool,
    mut on_event: impl FnMut(serde_json::Value) + Send + 'static,
    mut on_status: impl FnMut(crate::types::LiveStatus, Option<String>) + Send + 'static,
    mut should_stop: impl FnMut() -> bool + Send + 'static,
) {
    on_status(crate::types::LiveStatus::Connecting, None);
    let result = run_openblive_worker_inner(
        &identity_code,
        enable_ceve_server_heartbeat,
        &mut on_event,
        &mut on_status,
        &mut should_stop,
    );
    match result {
        Ok(()) => on_status(crate::types::LiveStatus::Disconnected, None),
        Err(error) => on_status(crate::types::LiveStatus::Error, Some(error.to_string())),
    }
}

fn run_openblive_worker_inner(
    identity_code: &str,
    enable_ceve_server_heartbeat: bool,
    on_event: &mut impl FnMut(serde_json::Value),
    on_status: &mut impl FnMut(crate::types::LiveStatus, Option<String>),
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<()> {
    tracing::debug!(
        identity_code_len = identity_code.trim().len(),
        enable_ceve_server_heartbeat,
        "starting OpenLive worker"
    );
    if identity_code.trim().is_empty() {
        bail!("OpenLive identity code is empty");
    }
    if should_stop() {
        return Ok(());
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .context("failed to create OpenLive HTTP client")?;
    let mut state = LiveWorkerState::default();

    let start_data = match start_app(&client, identity_code).context("failed to start OpenLive app")
    {
        Ok(start_data) => start_data,
        Err(_) if should_stop() => return Ok(()),
        Err(error) => return Err(error),
    };
    let game_id = parse_start_game_id(&start_data)?;
    state.set_game_id(game_id.clone());
    if should_stop() {
        return finish_openblive_session(&client, &mut state, Ok(()));
    }

    let result = parse_start_details(&start_data, game_id).and_then(|start| {
        run_openblive_session(
            &client,
            &start,
            enable_ceve_server_heartbeat,
            on_event,
            on_status,
            should_stop,
        )
    });
    finish_openblive_session(&client, &mut state, result)
}

fn run_openblive_session(
    client: &Client,
    start: &StartAppResponse,
    enable_ceve_server_heartbeat: bool,
    on_event: &mut impl FnMut(serde_json::Value),
    on_status: &mut impl FnMut(crate::types::LiveStatus, Option<String>),
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<()> {
    let Some(mut websocket) = connect_openlive_websocket(start, should_stop)? else {
        return Ok(());
    };
    on_status(crate::types::LiveStatus::Connected, None);

    let mut next_heartbeat = Instant::now();
    loop {
        if should_stop() {
            break;
        }

        if Instant::now() >= next_heartbeat {
            websocket
                .send(Message::binary(
                    OpenBlivePacket {
                        op: OP_HEARTBEAT,
                        body: Vec::new(),
                    }
                    .pack(),
                ))
                .context("failed to send OpenLive websocket heartbeat")?;
            heartbeat_app_once(&client, &start.game_id)
                .context("failed to send OpenLive app heartbeat")?;
            if enable_ceve_server_heartbeat {
                if let Err(error) = heartbeat_ceve_once(&client, &start.game_id) {
                    tracing::warn!("CEVE heartbeat failed: {error}");
                }
            }
            next_heartbeat = Instant::now() + Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
        }

        match websocket.read() {
            Ok(Message::Binary(bytes)) => {
                handle_websocket_binary(&bytes, on_event)?;
            }
            Ok(Message::Text(text)) => {
                let raw: Value = serde_json::from_str(&text)
                    .with_context(|| "failed to decode OpenLive text websocket message")?;
                on_event(raw);
            }
            Ok(Message::Ping(bytes)) => {
                websocket
                    .send(Message::Pong(bytes))
                    .context("failed to send OpenLive pong")?;
            }
            Ok(Message::Pong(_) | Message::Frame(_)) => {}
            Ok(Message::Close(_)) => {
                if should_stop() {
                    break;
                }
                bail!("OpenLive websocket closed unexpectedly");
            }
            Err(tungstenite::Error::Io(error)) if is_timeout(&error) => {}
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                if should_stop() {
                    break;
                }
                bail!("OpenLive websocket connection closed unexpectedly");
            }
            Err(error) => return Err(error).context("OpenLive websocket read failed"),
        }
    }

    let _ = websocket.close(None);
    Ok(())
}

fn finish_openblive_session(
    client: &Client,
    state: &mut LiveWorkerState,
    result: anyhow::Result<()>,
) -> anyhow::Result<()> {
    let cleanup_result = state.run_disconnect_cleanup(|game_id| {
        end_app_once(client, game_id).context("failed to end OpenLive app")
    });

    match (result, cleanup_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(error.context(format!(
            "OpenLive cleanup also failed after session error: {cleanup_error}"
        ))),
    }
}

fn connect_openlive_websocket(
    start: &StartAppResponse,
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<Option<WebSocket<MaybeTlsStream<TcpStream>>>> {
    tracing::debug!(
        game_id = %start.game_id,
        websocket_url = %start.websocket_url,
        auth_body_len = start.auth_body.len(),
        "connecting OpenLive websocket"
    );
    let Some(mut websocket) =
        connect_openlive_websocket_with_timeout(&start.websocket_url, should_stop)?
    else {
        return Ok(None);
    };
    set_read_timeout(websocket.get_mut(), Some(READ_TIMEOUT))?;
    websocket
        .send(Message::binary(
            OpenBlivePacket {
                op: OP_AUTH,
                body: start.auth_body.as_bytes().to_vec(),
            }
            .pack(),
        ))
        .context("failed to send OpenLive websocket auth packet")?;
    tracing::debug!(
        game_id = %start.game_id,
        websocket_url = %start.websocket_url,
        "sent OpenLive websocket auth packet"
    );

    loop {
        if should_stop() {
            let _ = websocket.close(None);
            return Ok(None);
        }

        match websocket.read() {
            Ok(Message::Binary(bytes)) => {
                tracing::debug!(
                    game_id = %start.game_id,
                    websocket_url = %start.websocket_url,
                    frame_len = bytes.len(),
                    "received OpenLive websocket auth frame"
                );
                if handle_auth_binary(&bytes)? {
                    tracing::debug!(
                        game_id = %start.game_id,
                        websocket_url = %start.websocket_url,
                        "OpenLive websocket auth completed"
                    );
                    return Ok(Some(websocket));
                }
            }
            Ok(Message::Ping(bytes)) => {
                websocket
                    .send(Message::Pong(bytes))
                    .context("failed to send OpenLive pong during auth")?;
            }
            Ok(Message::Close(_)) => bail!("OpenLive websocket closed before auth completed"),
            Ok(_) => {}
            Err(tungstenite::Error::Io(error)) if is_timeout(&error) => {}
            Err(error) => return Err(error).context("OpenLive websocket auth failed"),
        }
    }
}

fn connect_openlive_websocket_with_timeout(
    websocket_url: &str,
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<Option<WebSocket<MaybeTlsStream<TcpStream>>>> {
    if should_stop() {
        return Ok(None);
    }

    let request = websocket_url
        .into_client_request()
        .with_context(|| format!("invalid OpenLive websocket URL {websocket_url}"))?;
    let uri = request.uri();
    let mode = uri_mode(uri).context("unsupported OpenLive websocket URL scheme")?;
    let host = uri
        .host()
        .ok_or_else(|| anyhow!("OpenLive websocket URL has no host"))?;
    let connect_host = host.trim_start_matches('[').trim_end_matches(']');
    let port = uri.port_u16().unwrap_or(match mode {
        Mode::Plain => 80,
        Mode::Tls => 443,
    });
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let mut last_error = None;

    let addresses =
        resolve_socket_addrs_with_deadline(connect_host.to_string(), port, deadline, should_stop)
            .with_context(|| format!("failed to resolve OpenLive websocket host {host}"))?;
    let Some(addresses) = addresses else {
        return Ok(None);
    };

    for address in addresses {
        if should_stop() {
            return Ok(None);
        }
        let remaining = connect_budget_remaining(deadline)?;

        match TcpStream::connect_timeout(&address, remaining) {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                let remaining = connect_budget_remaining(deadline)?;
                stream.set_read_timeout(Some(remaining)).ok();
                stream.set_write_timeout(Some(remaining)).ok();
                if should_stop() {
                    return Ok(None);
                }
                let (websocket, _) = client_tls_with_config(request.clone(), stream, None, None)
                    .map_err(tungstenite_handshake_error)
                    .with_context(|| {
                        format!("failed to complete OpenLive websocket handshake {websocket_url}")
                    })?;
                if should_stop() {
                    return Ok(None);
                }
                return Ok(Some(websocket));
            }
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error
        .map(|error| anyhow!("failed to connect OpenLive websocket {websocket_url}: {error}"))
        .unwrap_or_else(|| anyhow!("OpenLive websocket host resolved to no addresses: {host}")))
}

fn connect_budget_remaining(deadline: Instant) -> anyhow::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| anyhow!("OpenLive websocket connect timed out"))
}

fn resolve_socket_addrs_with_deadline(
    host: String,
    port: u16,
    deadline: Instant,
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<Option<Vec<SocketAddr>>> {
    if should_stop() {
        return Ok(None);
    }
    let remaining = connect_budget_remaining(deadline)?;
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("voxui-openblive-dns".to_string())
        .spawn(move || {
            let result = (host.as_str(), port)
                .to_socket_addrs()
                .map(|addresses| addresses.collect::<Vec<_>>());
            let _ = sender.send(result);
        })
        .context("failed to spawn OpenLive DNS resolver")?;

    wait_for_resolution(receiver, deadline, remaining, should_stop)
}

fn wait_for_resolution(
    receiver: mpsc::Receiver<io::Result<Vec<SocketAddr>>>,
    deadline: Instant,
    mut remaining: Duration,
    should_stop: &mut impl FnMut() -> bool,
) -> anyhow::Result<Option<Vec<SocketAddr>>> {
    loop {
        if should_stop() {
            return Ok(None);
        }
        let wait = remaining.min(STOP_CHECK_INTERVAL);
        match receiver.recv_timeout(wait) {
            Ok(Ok(addresses)) => return Ok(Some(addresses)),
            Ok(Err(error)) => return Err(error.into()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("OpenLive DNS resolver exited without a result")
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                remaining = connect_budget_remaining(deadline)?;
            }
        }
    }
}

fn tungstenite_handshake_error(
    error: HandshakeError<tungstenite::ClientHandshake<MaybeTlsStream<TcpStream>>>,
) -> tungstenite::Error {
    match error {
        HandshakeError::Failure(error) => error,
        HandshakeError::Interrupted(_) => tungstenite::Error::Io(io::Error::new(
            io::ErrorKind::Interrupted,
            "OpenLive websocket handshake interrupted",
        )),
    }
}

pub fn handle_websocket_binary(
    bytes: &[u8],
    on_event: &mut impl FnMut(serde_json::Value),
) -> anyhow::Result<()> {
    for packet in unpack_packets(bytes)? {
        match packet.op {
            OP_MESSAGE => {
                let value: Value = serde_json::from_slice(&packet.body)
                    .context("failed to decode OpenLive event JSON")?;
                on_event(value);
            }
            OP_HEARTBEAT_REPLY | OP_AUTH_REPLY => {}
            op => tracing::debug!(op, "ignoring OpenLive websocket packet"),
        }
    }
    Ok(())
}

pub fn handle_auth_binary(bytes: &[u8]) -> anyhow::Result<bool> {
    let packets = unpack_packets(bytes)?;
    let mut saw_packet = false;
    for packet in packets {
        saw_packet = true;
        if packet.op != OP_AUTH_REPLY {
            continue;
        }
        validate_auth_reply(&packet.body)?;
        tracing::debug!("OpenLive websocket auth reply accepted");
        return Ok(true);
    }
    if saw_packet {
        tracing::debug!("OpenLive websocket auth frame did not contain an auth reply");
    }
    Ok(false)
}

fn validate_auth_reply(body: &[u8]) -> anyhow::Result<()> {
    if body.is_empty() {
        return Ok(());
    }

    let value: Value =
        serde_json::from_slice(body).context("failed to decode OpenLive auth reply JSON")?;
    let code = value
        .get("code")
        .or_else(|| value.pointer("/data/code"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if code == 0 {
        return Ok(());
    }

    let message = value
        .get("message")
        .or_else(|| value.get("msg"))
        .or_else(|| value.pointer("/data/message"))
        .and_then(Value::as_str)
        .unwrap_or("OpenLive websocket auth failed");
    bail!("{message}");
}

fn set_read_timeout(
    stream: &mut MaybeTlsStream<TcpStream>,
    timeout: Option<Duration>,
) -> anyhow::Result<()> {
    match stream {
        MaybeTlsStream::Plain(stream) => stream.set_read_timeout(timeout)?,
        MaybeTlsStream::Rustls(stream) => stream.sock.set_read_timeout(timeout)?,
        #[allow(unreachable_patterns)]
        _ => {}
    }
    Ok(())
}

fn is_timeout(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

fn sign_body(client: &Client, body: &Value) -> anyhow::Result<HeaderMap> {
    let compact_body = compact_json_body(body)?;
    let mut last_error = None;

    for url in SIGN_URLS {
        tracing::debug!(
            url,
            request_body = %compact_body,
            "requesting OpenLive signature"
        );
        let response = match client
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .body(compact_body.clone())
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(anyhow!(
                    "failed to request OpenLive signature from {url}: {error}"
                ));
                continue;
            }
        };

        let status = response.status();
        let response_text = response
            .text()
            .with_context(|| format!("failed to read OpenLive signature response from {url}"))?;
        tracing::debug!(
            url,
            status = %status,
            response = %response_text,
            "received OpenLive signature response"
        );

        if !status.is_success() {
            last_error = Some(anyhow!(
                "OpenLive signature server {url} returned HTTP {status}: {response_text}"
            ));
            continue;
        }

        let value: Value = serde_json::from_str(&response_text)
            .with_context(|| format!("failed to decode OpenLive signature response from {url}"))?;
        let headers = signed_headers_from_response(&value)
            .with_context(|| format!("invalid OpenLive signature response from {url}"));
        if let Ok(headers) = headers {
            let header_names: Vec<_> = headers.keys().map(|name| name.as_str()).collect();
            tracing::debug!(
                url,
                header_names = ?header_names,
                "parsed OpenLive signature response"
            );
            return Ok(headers);
        }
        return headers;
    }

    Err(last_error.unwrap_or_else(|| anyhow!("no OpenLive signature servers configured")))
}

fn signed_headers_from_response(value: &Value) -> anyhow::Result<HeaderMap> {
    let candidates = [
        value.get("headers"),
        value.pointer("/data/headers"),
        value.get("data"),
        Some(value),
    ];

    for candidate in candidates.into_iter().flatten() {
        if let Some(object) = candidate.as_object() {
            let mut headers = HeaderMap::new();
            for (key, value) in object {
                let Some(value) = value.as_str() else {
                    continue;
                };
                if !is_probable_header_name(key) {
                    continue;
                }
                let name = HeaderName::from_bytes(key.as_bytes())
                    .with_context(|| format!("invalid signed OpenLive header name {key}"))?;
                let value = HeaderValue::from_str(value)
                    .with_context(|| format!("invalid signed OpenLive header value for {key}"))?;
                headers.insert(name, value);
            }
            if !headers.is_empty() {
                return Ok(headers);
            }
        }
    }

    bail!("OpenLive signature response did not contain signed headers")
}

fn is_probable_header_name(key: &str) -> bool {
    key.to_ascii_lowercase().starts_with("x-")
        || key.eq_ignore_ascii_case("authorization")
        || key.eq_ignore_ascii_case("content-md5")
}

fn post_openlive(client: &Client, path: &str, body: &Value) -> anyhow::Result<Value> {
    let headers = sign_body(client, body)?;
    let compact_body = compact_json_body(body)?;
    let url = format!("{HOST}{path}");
    let header_names: Vec<_> = headers.keys().map(|name| name.as_str()).collect();
    tracing::debug!(
        path,
        url = %url,
        request_body = %compact_body,
        signed_header_names = ?header_names,
        "posting OpenLive API request"
    );
    let response = client
        .post(&url)
        .headers(headers)
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json")
        .body(compact_body)
        .send()
        .with_context(|| format!("failed to post OpenLive API {path}"))?;

    let status = response.status();
    let response_text = response
        .text()
        .with_context(|| format!("failed to read OpenLive API {path} response"))?;
    tracing::debug!(
        path,
        status = %status,
        response = %redact_openlive_response(path, &response_text),
        "received OpenLive API response"
    );

    if !status.is_success() {
        bail!("OpenLive API {path} returned HTTP {status}: {response_text}");
    }

    let value: Value = serde_json::from_str(&response_text)
        .with_context(|| format!("failed to decode OpenLive API {path} response"))?;
    let code = value.get("code").and_then(Value::as_i64).unwrap_or(0);
    if code != 0 {
        let message = value
            .get("message")
            .or_else(|| value.get("msg"))
            .and_then(Value::as_str)
            .unwrap_or("OpenLive API returned an error");
        bail!("OpenLive API {path} failed with code {code}: {message}; response={response_text}");
    }

    Ok(value.get("data").cloned().unwrap_or(Value::Null))
}

fn start_app(client: &Client, identity_code: &str) -> anyhow::Result<Value> {
    post_openlive(
        client,
        "/v2/app/start",
        &serde_json::json!({
            "code": identity_code,
            "app_id": APP_ID,
        }),
    )
}

fn parse_start_game_id(data: &Value) -> anyhow::Result<String> {
    required_string(
        data,
        &["/game_info/game_id", "/game_id"],
        "OpenLive start response missing game_id",
    )
}

fn parse_start_details(data: &Value, game_id: String) -> anyhow::Result<StartAppResponse> {
    let websocket_info = data.get("websocket_info").unwrap_or(&data);
    let auth_body = required_string_or_json(
        websocket_info,
        &["/auth_body"],
        "OpenLive start response missing websocket auth_body",
    )?;
    let websocket_url = websocket_url(websocket_info)
        .context("OpenLive start response missing websocket wss_link")?;
    tracing::debug!(
        game_id = %game_id,
        websocket_url = %websocket_url,
        auth_body_len = auth_body.len(),
        "parsed OpenLive start response"
    );

    Ok(StartAppResponse {
        game_id,
        auth_body,
        websocket_url,
    })
}

fn heartbeat_app_once(client: &Client, game_id: &str) -> anyhow::Result<()> {
    tracing::debug!(game_id, "sending OpenLive app heartbeat");
    post_openlive(
        client,
        "/v2/app/heartbeat",
        &serde_json::json!({ "game_id": game_id }),
    )
    .map(|_| ())
}

fn heartbeat_ceve_once(client: &Client, game_id: &str) -> anyhow::Result<()> {
    tracing::debug!(game_id, "sending CEVE heartbeat");
    let response = client
        .post(CEVE_HEARTBEAT_URL)
        .header(ACCEPT, "application/json")
        .json(&serde_json::json!({ "gameId": game_id }))
        .send()
        .context("failed to post CEVE heartbeat")?;
    if !response.status().is_success() {
        bail!("CEVE heartbeat returned HTTP {}", response.status());
    }
    Ok(())
}

fn end_app_once(client: &Client, game_id: &str) -> anyhow::Result<()> {
    tracing::debug!(game_id, "ending OpenLive app");
    post_openlive(
        client,
        "/v2/app/end",
        &serde_json::json!({ "game_id": game_id }),
    )
    .map(|_| ())
}

fn websocket_url(value: &Value) -> Option<String> {
    match value.get("wss_link") {
        Some(Value::String(url)) => Some(url.clone()),
        Some(Value::Array(urls)) => urls
            .iter()
            .filter_map(Value::as_str)
            .find(|url| !url.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

fn redact_openlive_response(path: &str, response_text: &str) -> String {
    if path != "/v2/app/start" {
        return response_text.to_string();
    }

    let Ok(mut value) = serde_json::from_str::<Value>(response_text) else {
        return response_text.to_string();
    };

    if let Some(auth_body) = value.pointer_mut("/data/websocket_info/auth_body") {
        *auth_body = Value::String("<redacted>".to_string());
    }

    serde_json::to_string(&value).unwrap_or_else(|_| response_text.to_string())
}

fn required_string(value: &Value, pointers: &[&str], message: &str) -> anyhow::Result<String> {
    for pointer in pointers {
        if let Some(text) = value.pointer(pointer).and_then(Value::as_str) {
            if !text.is_empty() {
                return Ok(text.to_string());
            }
        }
    }
    bail!("{message}")
}

fn required_string_or_json(
    value: &Value,
    pointers: &[&str],
    message: &str,
) -> anyhow::Result<String> {
    for pointer in pointers {
        if let Some(value) = value.pointer(pointer) {
            if let Some(text) = value.as_str() {
                if !text.is_empty() {
                    return Ok(text.to_string());
                }
            } else if !value.is_null() {
                return compact_json_body(value);
            }
        }
    }
    bail!("{message}")
}

#[derive(Serialize)]
#[serde(untagged)]
enum SortedJsonValue<'a> {
    Null,
    Bool(bool),
    Number(&'a serde_json::Number),
    String(&'a str),
    Array(Vec<SortedJsonValue<'a>>),
    Object(BTreeMap<&'a str, SortedJsonValue<'a>>),
}

impl<'a> From<&'a Value> for SortedJsonValue<'a> {
    fn from(value: &'a Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(*value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value),
            Value::Array(values) => Self::Array(values.iter().map(Self::from).collect()),
            Value::Object(values) => Self::Object(
                values
                    .iter()
                    .map(|(key, value)| (key.as_str(), Self::from(value)))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn start_response_parsing_captures_game_id_before_websocket_fields() {
        let data = json!({
            "game_info": {
                "game_id": "game-1"
            },
            "websocket_info": {}
        });

        let game_id = parse_start_game_id(&data).unwrap();
        let result = parse_start_details(&data, game_id.clone());

        assert_eq!(game_id, "game-1");
        assert!(result.is_err());
    }

    #[test]
    fn parsed_start_game_id_arms_cleanup_before_stop_return() {
        let data = json!({
            "game_info": {
                "game_id": "game-1"
            }
        });
        let mut state = LiveWorkerState::default();
        let game_id = parse_start_game_id(&data).unwrap();

        state.set_game_id(game_id);
        let command = state.mark_disconnect_requested();

        assert_eq!(
            command,
            Some(WorkerCommand::EndApp {
                game_id: "game-1".to_string()
            })
        );
    }

    #[test]
    fn wait_for_resolution_returns_none_when_stop_requested() {
        let (_sender, receiver) = mpsc::channel();
        let mut should_stop = || true;

        let result = wait_for_resolution(
            receiver,
            Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
            &mut should_stop,
        )
        .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn wait_for_resolution_returns_addresses_from_helper_thread() {
        let (sender, receiver) = mpsc::channel();
        let expected: SocketAddr = "127.0.0.1:80".parse().unwrap();
        sender.send(Ok(vec![expected])).unwrap();
        let mut should_stop = || false;

        let result = wait_for_resolution(
            receiver,
            Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
            &mut should_stop,
        )
        .unwrap();

        assert_eq!(result, Some(vec![expected]));
    }
}
