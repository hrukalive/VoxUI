use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Default)]
pub struct GeneratedAudioCache {
    audios: HashMap<String, GeneratedAudio>,
}

impl GeneratedAudioCache {
    pub fn insert(&mut self, id: String, audio: GeneratedAudio) {
        self.audios.insert(id, audio);
    }

    pub fn contains(&self, id: &str) -> bool {
        self.audios.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&GeneratedAudio> {
        self.audios.get(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<GeneratedAudio> {
        self.audios.remove(id)
    }
}
