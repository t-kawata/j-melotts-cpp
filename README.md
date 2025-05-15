# MeloTTS.cpp Japanese Support Research Repository

## Overview  
This repository facilitates Japanese language support research for **MeloTTS.cpp** through systematic comparison with the original **MeloTTS** codebase. The goal is to identify components requiring porting/translation to C++ while leveraging existing multilingual frameworks.

---

## Japanese Language Implementation Roadmap  
Based on codebase analysis and existing Chinese/English implementations ([^1][^2]), here are critical components for Japanese support:

### 1. **Language Module**  
- **Location**:  
  `src/language_modules/japanese.{h,cpp}`  
- **Implementation**:  
  Inherits from `AbstractLanguageModule` ([ref](https://github.com/apinge/MeloTTS.cpp/blob/main/src/language_modules/language_module_base.h#L27-L44))  
```

class Japanese : public AbstractLanguageModule {
// Required overrides:
std::string g2p(const std::string\& text) override;
std::string text_normalize(const std::string\& text) override;
std::map<std::string, int> symbol_to_id() override;
std::string get_language_name() override { return "JP"; }
};

```

### 2. **Text Normalization**  
- **Location**:  
`src/text_normalization/text_normalization_jp.{h,cpp}`  
- **Features**:  
- Japanese-specific handlers for numbers/abbreviations ([EN ref](https://github.com/apinge/MeloTTS.cpp/blob/main/src/text_normalization/text_normalization_eng.h#L24-L33))  
- Kanji-to-kana conversion logic

### 3. **Tokenization System**  
- **Components**:  
- **General Tokenizer**: Extend `Tokenizer` class ([ref](https://github.com/apinge/MeloTTS.cpp/blob/main/src/tokenizer.h#L27-L61)) with Japanese morphological analysis
- **BERT Tokenizer**: Update `OpenVinoTokenizer` to support Japanese models ([ref](https://github.com/apinge/MeloTTS.cpp/blob/main/src/openvino_tokenizer.h))

### 4. **Grapheme-to-Phoneme (G2P)**  
- **Implementation**:  
- Integrated in Japanese language module (similar to [Chinese G2P](https://github.com/apinge/MeloTTS.cpp/blob/main/src/language_modules/chinese_mix.h#L49-L53))  
- Potential use of `mini-bart-g2p` model with OpenVINO adaptation

### 5. **BERT Model Integration**  
- **Modifications**:  
```

// In bert.h constructor (L29-L37):
if (language == "JP") {
// Japanese-specific initialization
}

```
- **Feature Extraction**: Japanese-specific processing in hidden layer generation

### 6. **Phoneme Mapping**  
- **Structure**:  
Japanese-specific symbol-to-ID map mirroring [English implementation](https://github.com/apinge/MeloTTS.cpp/blob/main/src/language_modules/english.h#L43-L71)  
```

const std::map<std::string, int> JAPANESE_SYMBOLS = {
{"a", 1}, {"i", 2}, {"u", 3}, {"e", 4}, {"o", 5},
// ... 40+ Japanese phonemes
};

```

---

## Implementation Notes  
1. **Preliminary Support**:  
 - Tone handling exists in `language_module_base.h` ([L46](https://github.com/apinge/MeloTTS.cpp/blob/main/src/language_modules/language_module_base.h#L46))  
 - Language ID "JP" already mapped ([L66-L70](https://github.com/apinge/MeloTTS.cpp/blob/main/src/language_modules/language_module_base.h#L66-L70))

2. **Key Challenges**:  
 - Three writing systems: Hiragana, Katakana, Kanji  
 - Pitch accent vs. Chinese tonal system  
 - Agglutinative morphology requiring specialized tokenization

3. **Dependencies**:  
 Requires Japanese tokenizer library (equivalent to Jieba for Chinese) for effective morphological analysis.

---

## Repository Structure  
```

MeloTTS/          \# Original Python implementation (MyShell.ai)
MeloTTS.cpp/      \# C++ port (apinge)
├── src/
│   ├── language_modules/
│   │   ├── japanese.h       <-- New
│   │   └── japanese.cpp     <-- New
│   └── text_normalization/
│       └── text_normalization_jp.h  <-- New

```

---

This roadmap provides architectural guidance while preserving compatibility with OpenVINO deployment ([^1][^5][^6]). Contributors should reference both codebases and Japanese linguistic specifications during implementation.