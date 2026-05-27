use std::collections::HashMap;

use anyhow::{bail, Result};

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

    pub fn append(&mut self, id: String, samples: Vec<f32>, sample_rate: u32) -> Result<()> {
        match self.audios.get_mut(&id) {
            Some(audio) => {
                if audio.sample_rate != sample_rate {
                    bail!("sample rate changed for generated audio item {id}");
                }
                audio.samples.extend(samples);
            }
            None => {
                self.audios.insert(
                    id,
                    GeneratedAudio {
                        samples,
                        sample_rate,
                    },
                );
            }
        }
        Ok(())
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
