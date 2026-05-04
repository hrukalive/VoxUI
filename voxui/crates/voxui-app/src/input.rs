/// Text input state with cursor management.
#[derive(Debug, Default)]
pub struct InputState {
    pub text: String,
    pub cursor: usize,
    pub max_chars: usize,
    /// Set to true when input is rejected (e.g. max_chars exceeded). UI can use this for feedback.
    pub rejected: bool,
}

impl InputState {
    pub fn new(max_chars: usize) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            max_chars,
            rejected: false,
        }
    }

    pub fn insert(&mut self, ch: char) {
        if self.text.chars().count() >= self.max_chars {
            self.rejected = true;
            // Terminal bell
            let _ = crossterm::execute!(std::io::stdout(), crossterm::style::Print("\x07"));
            return;
        }
        self.rejected = false;
        let byte_pos = self.byte_offset(self.cursor);
        self.text.insert(byte_pos, ch);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let byte_pos = self.byte_offset(self.cursor);
            let next = self.byte_offset(self.cursor + 1);
            self.text.drain(byte_pos..next);
        }
    }

    pub fn delete(&mut self) {
        let len = self.text.chars().count();
        if self.cursor < len {
            let byte_pos = self.byte_offset(self.cursor);
            let next = self.byte_offset(self.cursor + 1);
            self.text.drain(byte_pos..next);
        }
    }

    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn move_right(&mut self) {
        let len = self.text.chars().count();
        if self.cursor < len {
            self.cursor += 1;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.chars().count();
    }

    pub fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    fn byte_offset(&self, char_idx: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_idx)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len())
    }
}
