#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Language {
    English,
    Chinese,
}

impl Default for Language {
    fn default() -> Self {
        Language::Chinese
    }
}

impl Language {
    pub fn display_name(self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Chinese => "中文",
        }
    }

    pub const ALL: [Language; 2] = [Language::English, Language::Chinese];
}

pub struct Strings {
    pub app_title: &'static str,
    pub tts_history: &'static str,
    pub input_placeholder: &'static str,
    pub settings_title: &'static str,
    pub settings_model: &'static str,
    pub settings_lora: &'static str,
    pub settings_backend: &'static str,
    pub settings_audio_host: &'static str,
    pub settings_audio_device: &'static str,
    pub settings_max_chars: &'static str,
    pub settings_dit_steps: &'static str,
    pub settings_language: &'static str,
    pub settings_apply: &'static str,
    pub settings_cancel: &'static str,
    pub settings_next: &'static str,
    pub settings_change: &'static str,
    pub status_loading: &'static str,
    pub status_ready: &'static str,
    pub status_error: &'static str,
    pub progress_generating: &'static str,
    pub progress_playing: &'static str,
    pub input_hint: &'static str,
    pub model_not_found_title: &'static str,
    pub model_not_found_msg: &'static str,
    pub model_not_found_hint: &'static str,
    pub model_not_found_error: &'static str,
    pub confirm: &'static str,
    pub none: &'static str,
}

const ENGLISH: Strings = Strings {
    app_title: "VoxUI",
    tts_history: "TTS History",
    input_placeholder: "> Type text and press Enter to send...",
    settings_title: "Settings",
    settings_model: "Model",
    settings_lora: "LoRA",
    settings_backend: "Backend",
    settings_audio_host: "Audio Host",
    settings_audio_device: "Audio Device",
    settings_max_chars: "Max Chars",
    settings_dit_steps: "Diffusion Steps",
    settings_language: "Language",
    settings_apply: "Apply",
    settings_cancel: "Cancel",
    settings_next: "Next",
    settings_change: "Change",
    status_loading: "Loading model...",
    status_ready: "Ready",
    status_error: "Error",
    progress_generating: "generating...",
    progress_playing: "playing...",
    input_hint: "Enter: send | F2: settings | Esc: quit",
    model_not_found_title: "Model Not Found",
    model_not_found_msg: "No model directory found. Please enter the path to your GGUF model folder:",
    model_not_found_hint: "The folder should contain manifest.json and model component files.",
    model_not_found_error: "manifest.json not found in this directory!",
    confirm: "OK",
    none: "None",
};

const CHINESE: Strings = Strings {
    app_title: "VoxUI",
    tts_history: "语音合成历史",
    input_placeholder: "> 输入文字按 Enter 发送...",
    settings_title: "设置",
    settings_model: "模型",
    settings_lora: "LoRA",
    settings_backend: "推理后端",
    settings_audio_host: "音频驱动",
    settings_audio_device: "音频设备",
    settings_max_chars: "最大字数",
    settings_dit_steps: "扩散步数",
    settings_language: "语言",
    settings_apply: "应用",
    settings_cancel: "取消",
    settings_next: "下一项",
    settings_change: "切换",
    status_loading: "正在加载模型...",
    status_ready: "就绪",
    status_error: "错误",
    progress_generating: "生成中...",
    progress_playing: "播放中...",
    input_hint: "Enter: 发送 | F2: 设置 | Esc: 退出",
    model_not_found_title: "未找到模型",
    model_not_found_msg: "未找到模型目录，请输入 GGUF 模型文件夹路径：",
    model_not_found_hint: "文件夹应包含 manifest.json 和模型组件文件。",
    model_not_found_error: "该目录下未找到 manifest.json！",
    confirm: "确定",
    none: "无",
};

pub fn get_strings(lang: Language) -> &'static Strings {
    match lang {
        Language::English => &ENGLISH,
        Language::Chinese => &CHINESE,
    }
}
