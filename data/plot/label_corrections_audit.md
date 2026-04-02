Prompt Injection Detection Dataset - Summary

  Overview

  A 667K-sample binary classification dataset for training a prompt injection detector in chat applications. Each sample is a text
  input labeled as benign (0) or injection (1), covering 19 distinct attack categories across real-world and synthetic sources. The
   dataset is formatted as a simplified single-token conversation:

  messages = [
      {"role": "system", "content": "You are a prompt injection detector. Reply only yes or no."},
      {"role": "user", "content": "<text to classify>"},
  ]
  # Expected response: "yes" or "no"

  Dataset Splits

  ┌────────────┬─────────┬─────────┬───────────┬────────────────┐
  │   Split    │ Samples │ Benign  │ Injection │ Injection Rate │
  ├────────────┼─────────┼─────────┼───────────┼────────────────┤
  │ Train      │ 548,769 │ 290,371 │   258,398 │     47.1%      │
  ├────────────┼─────────┼─────────┼───────────┼────────────────┤
  │ Validation │  78,395 │  41,487 │    36,908 │     47.1%      │
  ├────────────┼─────────┼─────────┼───────────┼────────────────┤
  │ Test       │  39,924 │  30,000 │     9,924 │     24.9%      │
  ├────────────┼─────────┼─────────┼───────────┼────────────────┤
  │ Total      │ 667,088 │ 361,858 │   305,230 │     45.8%      │
  └────────────┴─────────┴─────────┴───────────┴────────────────┘

  - Train/Val: Mixed raw data + balanced synthetic augmentation. 99.999% unique (4 duplicates out of 549K).
  - Test: Pure raw test splits from original HuggingFace datasets. Zero overlap with train/val guaranteed via cross-split
  deduplication.

  Attack Category Coverage (19 categories)

  ┌───────────────────────┬─────────────┬───────┬─────────────────────────────────────────────────────┐
  │       Category        │ Train Count │   %   │                     Description                     │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ jailbreak             │      75,008 │ 29.0% │ Developer mode, "no restrictions", unrestricted AI  │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ direct_injection      │      48,435 │ 18.7% │ "Ignore previous instructions", "forget rules"      │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ dan_roleplay          │      24,561 │  9.5% │ DAN/AIM/STAN persona adoption attacks               │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ multi_turn            │      17,508 │  6.8% │ Normal chat for N turns, then injection on turn N+1 │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ role_reversal         │      16,140 │  6.2% │ "I'm your admin/developer, disable filters"         │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ indirect_injection    │      16,059 │  6.2% │ Hidden instructions in emails, articles, documents  │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ payload_injection     │      15,459 │  6.0% │ Code execution, script injection, data exfiltration │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ leetspeak             │      15,308 │  5.9% │ 1gn0r3 4ll rul3s, multiple leet mappings            │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ social_engineering    │      10,401 │  4.0% │ Emotional manipulation, urgency, authority claims   │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ encoding_obfuscation  │       7,685 │  3.0% │ Base64, hex, ROT13, URL encoding, reversed text     │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ prompt_length         │       4,733 │  1.8% │ Extremely short/long prompts to overwhelm           │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ roleplay_variants     │       3,717 │  1.4% │ Character/fiction roleplay as attack vector         │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ control_prompt        │       2,688 │  1.0% │ System prompt probing, config extraction            │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ morse_code            │         536 │  0.2% │ Morse-encoded attack instructions                   │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ token_smuggling       │          46 │ <0.1% │ Zero-width chars, special token sequences           │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ case_manipulation     │          38 │ <0.1% │ ALL CAPS, aLtErNaTiNg case                          │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ multilingual_encoding │          32 │ <0.1% │ Non-English characters to evade filters             │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ combined_attack       │          24 │ <0.1% │ Multi-technique combinations                        │
  ├───────────────────────┼─────────────┼───────┼─────────────────────────────────────────────────────┤
  │ unicode_homoglyphs    │          20 │ <0.1% │ Unicode lookalike characters                        │
  └───────────────────────┴─────────────┴───────┴─────────────────────────────────────────────────────┘

  Data Sources

  ┌────────────────────────────────────────┬───────────────┬─────────────────────────────────────┐
  │                 Source                 │ Train Samples │                Type                 │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ benign-malicious-prompt-classification │       182,809 │ Raw (HuggingFace)                   │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-parametric                   │       147,243 │ Synthetic (combinatorial templates) │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ malicious-prompts                      │       110,104 │ Raw (HuggingFace / Gandalf)         │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ prompt-injection-safety                │        33,935 │ Raw (HuggingFace)                   │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-dan-roleplay                 │        24,475 │ Synthetic (persona adoption)        │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-multi-turn                   │        17,491 │ Synthetic (conversation attacks)    │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-leetspeak                    │        15,201 │ Synthetic (leet substitution)       │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-encoding                     │         7,203 │ Synthetic (encoding attacks)        │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ in-the-wild-regular                    │         3,700 │ Raw (TrustAIRLab)                   │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-roleplay                     │         3,613 │ Synthetic (character roleplay)      │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-control-prompt               │         1,602 │ Synthetic (probing/bypass)          │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-polite-bypass                │           416 │ Synthetic (subtle manipulation)     │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-prompt-extraction            │           369 │ Synthetic (output format exploit)   │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ synthetic-adversarial                  │           285 │ Synthetic (handcrafted novel)       │
  ├────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ in-the-wild-jailbreak                  │           323 │ Raw (TrustAIRLab)                   │
  └────────────────────────────────────────┴───────────────┴─────────────────────────────────────┘

  Mix: ~56% raw data from HuggingFace datasets, ~44% synthetic augmentation for balance and coverage.

  Quality Controls

  - Template deduplication: Per-template repetition capped at 20 instances to prevent Gandalf/PWNED pattern bias
  - Fuzzy dedup: Hash-based near-duplicate removal (word-set, prefix, suffix, char n-gram)
  - Cross-split dedup: Zero overlap guaranteed between train/val/test
  - Text cleaning: Unicode normalization, control char removal, HTML stripping, whitespace normalization
  - English dominance: >= 70% ASCII characters required
  - Length filter: 5-2000 characters

  Text Length Profile

  ┌────────┬───────────┬───────────┐
  │        │  Benign   │ Injection │
  ├────────┼───────────┼───────────┤
  │ Mean   │ 490 chars │ 224 chars │
  ├────────┼───────────┼───────────┤
  │ Median │ 391 chars │  92 chars │
  ├────────┼───────────┼───────────┤
  │ Range  │   5-2,000 │  6-11,676 │
  └────────┴───────────┴───────────┘

  Files

  ┌────────────────────────────────────┬──────────┬────────────────────────────────────┐
  │                File                │   Size   │            Description             │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/luna2_format/train.jsonl      │ 287.6 MB │ Training set (conversation format) │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/luna2_format/validation.jsonl │  41.1 MB │ Validation set                     │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/luna2_format/test.jsonl       │  29.4 MB │ Test set (raw data only)           │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/processed/dataset.json        │        — │ Split data (text + labels)         │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/processed/full_dataset.csv    │        — │ Full train dataset with metadata   │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/processed/metadata.json       │        — │ Config, stats, distributions       │
  ├────────────────────────────────────┼──────────┼────────────────────────────────────┤
  │ data/plots/*.png                   │  7 plots │ Distribution visualizations        │
  └────────────────────────────────────┴──────────┴────────────────────────────────────┘

  Design Decisions

  1. Simplified prompt: Minimal system prompt maximizes context window for input text. Fine-tuned models learn the task from data,
  not prompt description.
  2. Test = raw only: No synthetic data in test set for unbiased evaluation. Lower injection rate (24.9%) reflects real-world
  distribution.
  3. Category rebalancing: Dominant categories (direct_injection) capped to prevent single-pattern dominance. New attack vectors
  (DAN, multi-turn, leetspeak, encoding) synthetically boosted.
  4. Moderate imbalance accepted: Train is 47% injection / 53% benign - close enough for robust training without artificial
  oversampling.