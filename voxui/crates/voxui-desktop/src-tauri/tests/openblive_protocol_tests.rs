use voxui_desktop::openblive::{
    compact_json_body, handle_auth_binary, handle_websocket_binary, unpack_packet, unpack_packets,
    LiveWorkerState, OpenBlivePacket, WorkerCommand, APP_ID, CEVE_HEARTBEAT_URL,
    HEARTBEAT_INTERVAL_SECS, HOST, SIGN_URLS,
};

#[test]
fn openblive_constants_match_proven_provider_values() {
    assert_eq!(APP_ID, 1651388990835);
    assert_eq!(HOST, "https://live-open.biliapi.com");
    assert_eq!(SIGN_URLS[0], "https://soft.ceve-market.org/bopen/sign");
    assert_eq!(SIGN_URLS[1], "https://bopen.ceve-market.org/sign");
    assert_eq!(
        CEVE_HEARTBEAT_URL,
        "http://localhost.ceve-market.org:5218/heartbeat"
    );
    assert_eq!(HEARTBEAT_INTERVAL_SECS, 20);
}

#[test]
fn compact_json_body_matches_signing_requirement() {
    let body = compact_json_body(&serde_json::json!({
        "code": "ABC",
        "app_id": APP_ID
    }))
    .unwrap();

    assert_eq!(body, r#"{"app_id":1651388990835,"code":"ABC"}"#);
}

#[test]
fn packet_pack_round_trips_auth_body() {
    let packet = OpenBlivePacket {
        op: 7,
        body: br#"{"roomid":1}"#.to_vec(),
    };
    let packed = packet.pack();
    let decoded = unpack_packet(&packed).unwrap();

    assert_eq!(decoded.op, 7);
    assert_eq!(decoded.body, br#"{"roomid":1}"#);
}

#[test]
fn unpack_packets_decodes_concatenated_packets() {
    let first = OpenBlivePacket {
        op: 3,
        body: br#"{"count":1}"#.to_vec(),
    };
    let second = OpenBlivePacket {
        op: 8,
        body: br#"{"code":0}"#.to_vec(),
    };
    let mut batched = first.pack();
    batched.extend_from_slice(&second.pack());

    let decoded = unpack_packets(&batched).unwrap();

    assert_eq!(decoded, vec![first, second]);
}

#[test]
fn handle_websocket_binary_processes_all_messages_in_batched_frame() {
    let first = OpenBlivePacket {
        op: 5,
        body: br#"{"cmd":"LIVE_OPEN_PLATFORM_DM","data":{"msg":"one"}}"#.to_vec(),
    };
    let second = OpenBlivePacket {
        op: 5,
        body: br#"{"cmd":"LIVE_OPEN_PLATFORM_DM","data":{"msg":"two"}}"#.to_vec(),
    };
    let mut batched = first.pack();
    batched.extend_from_slice(&second.pack());
    let mut events = Vec::new();

    handle_websocket_binary(&batched, &mut |event| events.push(event)).unwrap();

    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["data"]["msg"], "one");
    assert_eq!(events[1]["data"]["msg"], "two");
}

#[test]
fn handle_auth_binary_accepts_auth_reply_after_heartbeat_reply_in_batched_frame() {
    let heartbeat = OpenBlivePacket {
        op: 3,
        body: br#"{"count":1}"#.to_vec(),
    };
    let auth = OpenBlivePacket {
        op: 8,
        body: br#"{"code":0}"#.to_vec(),
    };
    let mut batched = heartbeat.pack();
    batched.extend_from_slice(&auth.pack());

    assert!(handle_auth_binary(&batched).unwrap());
}

#[test]
fn unpack_packet_rejects_input_shorter_than_header() {
    assert!(unpack_packet(&[0; 15]).is_err());
}

#[test]
fn unpack_packet_rejects_declared_length_smaller_than_actual() {
    let mut packed = OpenBlivePacket {
        op: 7,
        body: br#"{"roomid":1}"#.to_vec(),
    }
    .pack();
    packed[0..4].copy_from_slice(&16_u32.to_be_bytes());

    assert!(unpack_packet(&packed).is_err());
}

#[test]
fn unpack_packet_rejects_declared_length_larger_than_actual() {
    let mut packed = OpenBlivePacket {
        op: 7,
        body: br#"{"roomid":1}"#.to_vec(),
    }
    .pack();
    let declared_len = u32::try_from(packed.len() + 1).unwrap();
    packed[0..4].copy_from_slice(&declared_len.to_be_bytes());

    assert!(unpack_packet(&packed).is_err());
}

#[test]
fn unpack_packet_rejects_invalid_header_length() {
    let mut packed = OpenBlivePacket {
        op: 7,
        body: br#"{"roomid":1}"#.to_vec(),
    }
    .pack();
    packed[4..6].copy_from_slice(&15_u16.to_be_bytes());

    assert!(unpack_packet(&packed).is_err());
}

#[test]
fn worker_state_disconnect_is_idempotent() {
    let mut state = LiveWorkerState::for_test(Some("game-1".to_string()));

    assert_eq!(
        state.mark_disconnect_requested(),
        Some(WorkerCommand::EndApp {
            game_id: "game-1".to_string()
        })
    );
    assert_eq!(state.mark_disconnect_requested(), None);
}

#[test]
fn worker_state_disconnect_cleanup_runs_once() {
    let mut state = LiveWorkerState::for_test(Some("game-1".to_string()));
    let mut cleaned_game_ids = Vec::new();

    state
        .run_disconnect_cleanup(|game_id| {
            cleaned_game_ids.push(game_id.to_string());
            Ok::<(), ()>(())
        })
        .unwrap();
    state
        .run_disconnect_cleanup(|game_id| {
            cleaned_game_ids.push(game_id.to_string());
            Ok::<(), ()>(())
        })
        .unwrap();

    assert_eq!(cleaned_game_ids, vec!["game-1".to_string()]);
}
