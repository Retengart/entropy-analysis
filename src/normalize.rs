use anyhow::{bail, Result};

use crate::cli::AlphabetChoice;
use crate::model::Normalization;

pub fn build_normalization(
    choice: AlphabetChoice,
    custom: Option<&str>,
    keep_yo: bool,
    keep_j: bool,
    min_token_len: usize,
) -> Result<Normalization> {
    let letters: Vec<char> = match choice {
        AlphabetChoice::Rus28 => rus28_alphabet(),
        AlphabetChoice::Rus33 => rus33_alphabet(),
        AlphabetChoice::Custom => {
            let s = custom.unwrap_or("");
            if s.is_empty() { bail!("--custom-alphabet must be provided for alphabet=custom"); }
            unique_chars_preserve_order(s)
        }
    };
    Ok(Normalization { alphabet: letters, rus28_keep_yo: keep_yo, rus28_keep_j: keep_j, min_token_len })
}

fn unique_chars_preserve_order(s: &str) -> Vec<char> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for ch in s.chars() {
        if !seen.contains(&ch) { seen.insert(ch); out.push(ch); }
    }
    out
}

pub fn rus28_alphabet() -> Vec<char> {
    // 28 letters per the methodic (no ё, й, ъ, ь)
    "абвгдежзиклмнопрстуфхцчшщыэюя".chars().collect()
}

pub fn rus33_alphabet() -> Vec<char> {
    // Standard Russian alphabet with ё, й, ъ, ы, ь
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя".chars().collect()
}


