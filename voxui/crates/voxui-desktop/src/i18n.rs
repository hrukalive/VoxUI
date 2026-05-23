#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiLanguage {
    Chinese,
    English,
}

#[derive(Debug, Clone, Copy)]
pub struct Labels {
    pub title: &'static str,
    pub subtitle: &'static str,
    pub load: &'static str,
    pub generate: &'static str,
    pub settings: &'static str,
    pub model: &'static str,
    pub input_placeholder: &'static str,
    pub history_empty: &'static str,
    pub cancel: &'static str,
    pub play: &'static str,
    pub stop: &'static str,
    pub regenerate: &'static str,
}

pub fn labels(language: UiLanguage) -> Labels {
    match language {
        UiLanguage::Chinese => Labels {
            title: "焓言焓语",
            subtitle: "AhanSays",
            load: "加载",
            generate: "生成",
            settings: "设置",
            model: "模型",
            input_placeholder: "输入要合成的文字...",
            history_empty: "暂无生成记录",
            cancel: "取消",
            play: "播放",
            stop: "停止",
            regenerate: "重新生成",
        },
        UiLanguage::English => Labels {
            title: "AhanSays",
            subtitle: "焓言焓语",
            load: "Load",
            generate: "Generate",
            settings: "Settings",
            model: "Model",
            input_placeholder: "Enter text to synthesize...",
            history_empty: "No generation history yet",
            cancel: "Cancel",
            play: "Play",
            stop: "Stop",
            regenerate: "Regenerate",
        },
    }
}
