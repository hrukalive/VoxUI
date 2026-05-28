use serde_json::json;
use voxui_desktop::live::{
    parse_live_event, render_suggestion, switch_text, LiveLanguage, SuggestionMode,
};
use voxui_desktop::types::{AppConfig, LiveMessageKind, ReplacementRule, TemplateConfig};

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
    assert_eq!(
        live.replacement_rules,
        vec![
            ReplacementRule {
                enabled: true,
                from: "我的".to_string(),
                to: "你的".to_string(),
            },
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
    assert_eq!(
        templates.superchat_zh,
        "感谢{mapped_uname}的醒目留言：{message}"
    );
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
