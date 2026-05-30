use std::fs;

use serde_json::json;
use tempfile::TempDir;
use voxui_desktop::live::{
    parse_live_event, render_suggestion, switch_text, LiveLanguage, SuggestionMode,
};
use voxui_desktop::types::{
    AppConfig, AutoGenMode, LanguageMode, LiveConfigPatch, LiveMessageKind, LiveStatus,
    ReplacementRule, SendMode, TemplateConfig,
};

#[test]
fn live_config_defaults_match_bilibili_monitor_spec() {
    let live = AppConfig::default().live;

    assert_eq!(live.identity_code, "");
    assert!(!live.enable_ceve_server_heartbeat);
    assert!(live.show_danmu);
    assert!(live.show_gifts);
    assert!(live.show_superchats);
    assert!(live.show_guards);
    assert!(!live.show_likes);
    assert!(live.show_enters);
    assert_eq!(live.send_mode, SendMode::Manual);
    assert_eq!(live.auto_gen_mode, AutoGenMode::None);
    assert!(!live.auto_gen_danmu);
    assert!(live.auto_gen_gifts);
    assert!(live.auto_gen_superchats);
    assert!(live.auto_gen_guards);
    assert!(!live.auto_gen_likes);
    assert!(live.auto_gen_enters);
    assert_eq!(
        live.replacement_rules,
        vec![
            ReplacementRule {
                enabled: true,
                from: "我".to_string(),
                to: "你".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "I".to_string(),
                to: "you".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "me".to_string(),
                to: "you".to_string(),
            },
            ReplacementRule {
                enabled: true,
                from: "my".to_string(),
                to: "your".to_string(),
            },
        ]
    );
    assert!(live.mapped_unames.is_empty());
    assert!(live.original_unames.is_empty());
    assert_eq!(LiveMessageKind::Gift.is_paid(), true);
    assert_eq!(LiveMessageKind::Danmu.is_paid(), false);
}

#[test]
fn old_config_json_deserializes_with_live_defaults() {
    let decoded: AppConfig = serde_json::from_str(r#"{ "max_input_chars": 123 }"#).unwrap();

    assert_eq!(decoded.max_input_chars, 123);
    assert!(decoded.live.show_danmu);
    assert_eq!(decoded.live.send_mode, SendMode::Manual);
    assert_eq!(decoded.live.auto_gen_mode, AutoGenMode::None);
    assert!(!decoded.live.auto_gen_danmu);
    assert!(decoded.live.auto_gen_gifts);
    assert!(!decoded.live.enable_ceve_server_heartbeat);
}

#[test]
fn live_template_config_contains_all_message_templates() {
    let templates = TemplateConfig::default();

    assert_eq!(templates.danmu, "{msg}");
    assert_eq!(
        templates.gift_zh,
        "感谢{mapped_uname}送出的{gift_num}个{gift_name}"
    );
    assert_eq!(
        templates.gift_en,
        "Thank you {mapped_uname} for {gift_num} {gift_name}"
    );
    assert_eq!(templates.superchat_zh, "感谢{mapped_uname}的SC：{message}");
    assert_eq!(
        templates.superchat_en,
        "Thank you {mapped_uname} for the superchat saying {message}"
    );
    assert_eq!(templates.guard_zh, "感谢{mapped_uname}开通的{guard_label}");
    assert_eq!(
        templates.guard_en,
        "Thank you {mapped_uname} for joining as {guard_label}"
    );
    assert_eq!(templates.like_zh, "感谢{mapped_uname}给直播间点赞");
    assert_eq!(
        templates.like_en,
        "Thank you {mapped_uname} for liking the stream"
    );
    assert_eq!(templates.enter_zh, "欢迎{mapped_uname}进入直播间");
    assert_eq!(
        templates.enter_en,
        "Hi {mapped_uname}, welcome to the stream"
    );
}

#[test]
fn danmu_skips_emote_only_and_cleans_text() {
    let emote = json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": {
            "dm_type": 1,
            "open_id": "u1",
            "uname": "Alice",
            "msg": "[dog]"
        }
    });
    assert!(parse_live_event(emote).unwrap().is_none());

    let text = json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": {
            "dm_type": 0,
            "open_id": "u1",
            "uname": "Alice",
            "msg": "hello [dog]  world"
        }
    });
    let event = parse_live_event(text).unwrap().unwrap();

    assert_eq!(
        render_suggestion(
            &event,
            &AppConfig::default().live,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        Some("hello.world".to_string())
    );
}

#[test]
fn danmu_returns_none_when_cleanup_removes_all_content() {
    let emote_text = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": {
            "dm_type": 0,
            "open_id": "u1",
            "uname": "Alice",
            "msg": "[dog]"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &emote_text,
            &AppConfig::default().live,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        None
    );

    let whitespace = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": {
            "dm_type": 0,
            "open_id": "u1",
            "uname": "Alice",
            "msg": "   \t  "
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &whitespace,
            &AppConfig::default().live,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        None
    );
}

#[test]
fn gift_skips_unpaid_and_renders_paid_with_mapped_name() {
    let unpaid = json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": {
            "paid": false,
            "open_id": "u1",
            "uname": "Alice",
            "gift_name": "花",
            "gift_num": 1
        }
    });
    assert!(parse_live_event(unpaid).unwrap().is_none());

    let paid = json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": {
            "paid": true,
            "open_id": "u1",
            "uname": "Alice",
            "gift_name": "花",
            "gift_num": 2
        }
    });
    let mut config = AppConfig::default().live;
    config
        .mapped_unames
        .insert("u1".to_string(), "A酱".to_string());
    let event = parse_live_event(paid).unwrap().unwrap();

    assert_eq!(
        render_suggestion(
            &event,
            &config,
            LiveLanguage::Chinese,
            SuggestionMode::Normal,
        ),
        Some("感谢A酱送出的2个花".to_string())
    );
}

#[test]
fn template_rendering_does_not_reprocess_inserted_values() {
    let mut config = AppConfig::default().live;
    config
        .mapped_unames
        .insert("u2".to_string(), "{message}".to_string());
    let superchat = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
        "data": {
            "open_id": "u2",
            "uname": "Bob",
            "message": "加油"
        }
    }))
    .unwrap()
    .unwrap();

    assert_eq!(
        render_suggestion(
            &superchat,
            &config,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        Some("Thank you {message} for the superchat saying 加油".to_string())
    );
}

#[test]
fn replacement_applies_to_message_body_before_template_rendering() {
    let mut config = AppConfig::default().live;
    config.replacement_rules = vec![ReplacementRule {
        enabled: true,
        from: "cat".to_string(),
        to: "dog".to_string(),
    }];

    config.templates.danmu = "cat template says {msg}".to_string();
    let danmu = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": {
            "dm_type": 0,
            "open_id": "u1",
            "uname": "Alice",
            "msg": "catnip"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &danmu,
            &config,
            LiveLanguage::English,
            SuggestionMode::Switch,
        ),
        Some("cat template says dognip".to_string())
    );

    config.templates.superchat_en = "cat template says {message}".to_string();
    let superchat = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
        "data": {
            "open_id": "u2",
            "uname": "Bob",
            "message": "cat"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &superchat,
            &config,
            LiveLanguage::English,
            SuggestionMode::Switch,
        ),
        Some("cat template says dog".to_string())
    );
}

#[test]
fn superchat_guard_like_and_enter_render_templates() {
    let config = AppConfig::default().live;

    let superchat = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SUPER_CHAT",
        "data": {
            "open_id": "u2",
            "uname": "Bob",
            "message": "加油"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &superchat,
            &config,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        Some("Thank you Bob for the superchat saying 加油".to_string())
    );

    let guard = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_GUARD",
        "data": {
            "guard_level": 3,
            "user_info": {
                "open_id": "u2",
                "uname": "Bob"
            }
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &guard,
            &config,
            LiveLanguage::Chinese,
            SuggestionMode::Normal,
        ),
        Some("感谢Bob开通的舰长".to_string())
    );

    let mut likes_config = AppConfig::default().live;
    likes_config.show_likes = true;
    let like = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIKE",
        "data": {
            "open_id": "u2",
            "uname": "Bob"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &like,
            &likes_config,
            LiveLanguage::Chinese,
            SuggestionMode::Normal,
        ),
        Some("感谢Bob给直播间点赞".to_string())
    );

    let enter = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER",
        "data": {
            "open_id": "u2",
            "uname": "Bob"
        }
    }))
    .unwrap()
    .unwrap();
    assert_eq!(
        render_suggestion(
            &enter,
            &config,
            LiveLanguage::English,
            SuggestionMode::Normal,
        ),
        Some("Hi Bob, welcome to the stream".to_string())
    );
}

#[test]
fn switch_text_applies_enabled_replacement_rules() {
    let mut config = AppConfig::default().live;
    let i_rule = config
        .replacement_rules
        .iter_mut()
        .find(|rule| rule.from == "I")
        .unwrap();
    i_rule.enabled = false;

    assert_eq!(
        switch_text("我的猫 and I like my chair", &config),
        "你的猫 and I like your chair"
    );
}

#[test]
fn switch_text_ignores_empty_replacement_sources() {
    let mut config = AppConfig::default().live;
    config.replacement_rules.push(ReplacementRule {
        enabled: true,
        from: String::new(),
        to: "x".to_string(),
    });

    assert_eq!(switch_text("abc", &config), "abc");
}

#[test]
fn app_auto_period_applies_to_live_monitor_items_and_suggestions() {
    let mut live = AppConfig::default().live;
    live.replacement_rules = vec![ReplacementRule {
        enabled: true,
        from: "cat".to_string(),
        to: "dog".to_string(),
    }];
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_DM",
        "data": { "msg": "cat", "open_id": "u1", "uname": "Alice" }
    }))
    .unwrap()
    .unwrap();

    let mut with_period = voxui_desktop::app_core::AppCore::from_config(AppConfig {
        language: LanguageMode::English,
        auto_period: true,
        live: live.clone(),
        ..AppConfig::default()
    })
    .unwrap();
    let with_period_id = with_period.add_live_event_for_test(event.clone()).unwrap();
    assert_eq!(
        with_period
            .live_snapshot_for_test(LiveLanguage::English)
            .items[0]
            .suggestion,
        "cat."
    );
    assert_eq!(
        with_period
            .live_suggestion_for_item(
                &with_period_id,
                LiveLanguage::English,
                SuggestionMode::Switch
            )
            .as_deref(),
        Some("dog.")
    );

    let mut without_period = voxui_desktop::app_core::AppCore::from_config(AppConfig {
        language: LanguageMode::English,
        auto_period: false,
        live,
        ..AppConfig::default()
    })
    .unwrap();
    let without_period_id = without_period.add_live_event_for_test(event).unwrap();
    assert_eq!(
        without_period
            .live_snapshot_for_test(LiveLanguage::English)
            .items[0]
            .suggestion,
        "cat"
    );
    assert_eq!(
        without_period
            .live_suggestion_for_item(
                &without_period_id,
                LiveLanguage::English,
                SuggestionMode::Switch
            )
            .as_deref(),
        Some("dog")
    );
}

#[test]
fn adding_live_event_initializes_name_mapping_and_recomputes_after_patch() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "花", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    })).unwrap().unwrap();

    let item_id = core.add_live_event_for_test(event).unwrap();
    let first = core.live_snapshot_for_test(LiveLanguage::Chinese);
    assert_eq!(
        first.config.original_unames.get("u1").map(String::as_str),
        Some("Alice")
    );
    assert_eq!(first.items[0].suggestion, "感谢Alice送出的2个花。");

    core.apply_live_patch(LiveConfigPatch {
        mapped_unames: Some(
            [("u1".to_string(), "A酱".to_string())]
                .into_iter()
                .collect(),
        ),
        ..LiveConfigPatch::default()
    })
    .unwrap();
    let second = core.live_snapshot_for_test(LiveLanguage::Chinese);

    assert_eq!(second.items[0].id, item_id);
    assert_eq!(second.items[0].suggestion, "感谢A酱送出的2个花。");
}

#[test]
fn adding_first_seen_live_event_does_not_configure_mapped_uname() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u-new", "uname": "Alice" }
    }))
    .unwrap()
    .unwrap();

    core.add_live_event_for_test(event).unwrap();
    let snapshot = core.live_snapshot_for_test(LiveLanguage::English);

    assert_eq!(
        snapshot
            .config
            .original_unames
            .get("u-new")
            .map(String::as_str),
        Some("Alice")
    );
    assert!(
        !snapshot.config.mapped_unames.contains_key("u-new"),
        "new viewers should not be treated as streamer-configured mapped names"
    );
    assert_eq!(snapshot.items[0].mapped_uname, "Alice");
    assert_eq!(
        snapshot.items[0].suggestion,
        "Thank you Alice for 1 flower."
    );
}

#[test]
fn live_patch_can_acknowledge_current_uname_for_existing_mapping() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u1", "uname": "AliceNew" }
    }))
    .unwrap()
    .unwrap();

    core.add_live_event_for_test(event).unwrap();
    core.apply_live_patch(LiveConfigPatch {
        mapped_unames: Some(
            [("u1".to_string(), "A-chan".to_string())]
                .into_iter()
                .collect(),
        ),
        original_unames: Some(
            [("u1".to_string(), "AliceNew".to_string())]
                .into_iter()
                .collect(),
        ),
        ..LiveConfigPatch::default()
    })
    .unwrap();

    let snapshot = core.live_snapshot_for_test(LiveLanguage::English);
    assert_eq!(
        snapshot
            .config
            .original_unames
            .get("u1")
            .map(String::as_str),
        Some("AliceNew")
    );
    assert_eq!(
        snapshot.config.mapped_unames.get("u1").map(String::as_str),
        Some("A-chan")
    );
    assert_eq!(
        snapshot.items[0].suggestion,
        "Thank you A-chan for 1 flower."
    );
}

#[test]
fn mapped_uname_patch_acknowledges_latest_live_uname() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let first = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u1", "uname": "Alice" }
    }))
    .unwrap()
    .unwrap();
    let renamed = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u1", "uname": "AliceNew" }
    }))
    .unwrap()
    .unwrap();

    core.add_live_event_for_test(first).unwrap();
    core.add_live_event_for_test(renamed).unwrap();
    core.apply_live_patch(LiveConfigPatch {
        mapped_unames: Some(
            [("u1".to_string(), "A-chan".to_string())]
                .into_iter()
                .collect(),
        ),
        ..LiveConfigPatch::default()
    })
    .unwrap();

    let snapshot = core.live_snapshot_for_test(LiveLanguage::English);
    assert_eq!(
        snapshot
            .config
            .original_unames
            .get("u1")
            .map(String::as_str),
        Some("AliceNew")
    );
}

#[test]
fn live_snapshot_filters_likes_by_default_until_enabled() {
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    let like = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIKE",
        "data": { "open_id": "u2", "uname": "Bob", "like_count": 1 }
    }))
    .unwrap()
    .unwrap();

    core.add_live_event_for_test(like).unwrap();
    assert!(core
        .live_snapshot_for_test(LiveLanguage::Chinese)
        .items
        .is_empty());

    core.apply_live_patch(LiveConfigPatch {
        show_likes: Some(true),
        ..LiveConfigPatch::default()
    })
    .unwrap();

    assert_eq!(
        core.live_snapshot_for_test(LiveLanguage::Chinese)
            .items
            .len(),
        1
    );
}

#[test]
fn live_status_defaults_to_disconnected() {
    let core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();

    assert_eq!(core.live_status_for_test(), LiveStatus::Disconnected);
}

#[test]
fn setting_live_status_uses_configured_live_language() {
    let mut live = AppConfig::default().live;
    live.templates.gift_zh = "ZH {mapped_uname} {gift_num} {gift_name}".to_string();
    live.templates.gift_en = "EN {mapped_uname} {gift_num} {gift_name}".to_string();
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig {
        language: LanguageMode::Chinese,
        live,
        ..AppConfig::default()
    })
    .unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "èŠ±", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    })).unwrap().unwrap();
    core.add_live_event_for_test(event).unwrap();

    let snapshot = core.set_live_status(LiveStatus::Connected, Some("ready".to_string()));

    assert_eq!(snapshot.status, LiveStatus::Connected);
    assert_eq!(snapshot.status_message.as_deref(), Some("ready"));
    assert_eq!(snapshot.items[0].suggestion, "ZH Alice 2 èŠ±。");
}

#[test]
fn applying_live_patch_returns_snapshot_in_configured_live_language() {
    let mut live = AppConfig::default().live;
    live.templates.gift_zh = "ZH {mapped_uname} {gift_num} {gift_name}".to_string();
    live.templates.gift_en = "EN {mapped_uname} {gift_num} {gift_name}".to_string();
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig {
        language: LanguageMode::Chinese,
        live,
        ..AppConfig::default()
    })
    .unwrap();
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "èŠ±", "gift_num": 2, "open_id": "u1", "uname": "Alice" }
    })).unwrap().unwrap();
    core.add_live_event_for_test(event).unwrap();

    let snapshot = core
        .apply_live_patch(LiveConfigPatch {
            show_gifts: Some(true),
            ..LiveConfigPatch::default()
        })
        .unwrap();

    assert_eq!(snapshot.items[0].suggestion, "ZH Alice 2 èŠ±。");
}

#[test]
fn adding_live_event_persists_only_when_name_mapping_changes() {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("voxui_config.json");
    let empty_mapping_path = temp.path().join("empty_mapping_config.json");
    let mut config = AppConfig::default();
    config
        .live
        .original_unames
        .insert("u-existing-empty".to_string(), String::new());
    config
        .live
        .mapped_unames
        .insert("u-existing-empty".to_string(), String::new());
    let mut empty_mapping_core = voxui_desktop::app_core::AppCore::from_config(config).unwrap();
    empty_mapping_core.set_config_path(empty_mapping_path.clone());

    let replaces_empty_mapping = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u-existing-empty", "uname": "Alice" }
    })).unwrap().unwrap();
    empty_mapping_core
        .add_live_event_for_test(replaces_empty_mapping)
        .unwrap();

    let replaced_snapshot = empty_mapping_core.live_snapshot_for_test(LiveLanguage::English);
    let replaced_saved = voxui_desktop::config::load_config(&empty_mapping_path).unwrap();
    assert_eq!(
        replaced_snapshot
            .config
            .original_unames
            .get("u-existing-empty")
            .map(String::as_str),
        Some("Alice")
    );
    assert_eq!(
        replaced_saved
            .live
            .original_unames
            .get("u-existing-empty")
            .map(String::as_str),
        Some("Alice")
    );
    assert_eq!(
        replaced_snapshot
            .config
            .mapped_unames
            .get("u-existing-empty")
            .map(String::as_str),
        Some("Alice")
    );
    assert!(empty_mapping_path.exists());

    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    core.set_config_path(config_path.clone());

    let empty_name_event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_LIKE",
        "data": { "open_id": "u-empty", "uname": "" }
    }))
    .unwrap()
    .unwrap();
    core.add_live_event_for_test(empty_name_event).unwrap();

    assert!(!config_path.exists());
    assert!(core
        .live_snapshot_for_test(LiveLanguage::English)
        .config
        .original_unames
        .is_empty());

    let named_event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u1", "uname": "Alice" }
    })).unwrap().unwrap();
    core.add_live_event_for_test(named_event.clone()).unwrap();
    assert!(config_path.exists());
    let saved = voxui_desktop::config::load_config(&config_path).unwrap();
    assert_eq!(
        saved.live.original_unames.get("u1").map(String::as_str),
        Some("Alice")
    );
    assert!(
        !saved.live.mapped_unames.contains_key("u1"),
        "first-seen users should not be persisted as configured mapped names"
    );

    fs::write(&config_path, "sentinel").unwrap();
    core.add_live_event_for_test(named_event).unwrap();

    assert_eq!(fs::read_to_string(&config_path).unwrap(), "sentinel");
}

#[test]
fn mapped_uname_patch_persists_acknowledged_original_uname() {
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("voxui_config.json");
    let mut core = voxui_desktop::app_core::AppCore::from_config(AppConfig::default()).unwrap();
    core.set_config_path(config_path.clone());
    let event = parse_live_event(json!({
        "cmd": "LIVE_OPEN_PLATFORM_SEND_GIFT",
        "data": { "paid": true, "gift_name": "flower", "gift_num": 1, "open_id": "u1", "uname": "AliceNew" }
    }))
    .unwrap()
    .unwrap();

    core.add_live_event_for_test(event).unwrap();
    core.apply_live_patch(LiveConfigPatch {
        mapped_unames: Some(
            [("u1".to_string(), "A-chan".to_string())]
                .into_iter()
                .collect(),
        ),
        ..LiveConfigPatch::default()
    })
    .unwrap();

    let saved = voxui_desktop::config::load_config(&config_path).unwrap();
    assert_eq!(
        saved.live.original_unames.get("u1").map(String::as_str),
        Some("AliceNew")
    );
    assert_eq!(
        saved.live.mapped_unames.get("u1").map(String::as_str),
        Some("A-chan")
    );
}
