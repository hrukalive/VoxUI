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
