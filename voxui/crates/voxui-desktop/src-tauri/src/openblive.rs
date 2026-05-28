use std::collections::BTreeMap;

use anyhow::{anyhow, bail};
use serde::Serialize;
use serde_json::Value;

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

pub fn compact_json_body(value: &Value) -> anyhow::Result<String> {
    serde_json::to_string(&SortedJsonValue::from(value))
        .map_err(|error| anyhow!("failed to serialize compact OpenLive JSON body: {error}"))
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
