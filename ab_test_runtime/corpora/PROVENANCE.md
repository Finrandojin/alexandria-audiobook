# Corpora used by the non-English attribution experiments

These files are **not** in version control — they are downloadable from the
sources below, and 11 MB of public-domain text does not belong in a repository
whose history is already several gigabytes. What is committed is this file, so
the data can be reconstructed.

They previously lived in a Claude session scratchpad whose path embedded a
session UUID, which meant `chinese_attribution.py`, `quote_aware_chunking.py`
and `japanese_quote_robustness.py` referenced a directory that would disappear
when that session was cleaned. Moved here on 2026-08-04.

Override the location with `ALEXANDRIA_CHINESE_CORPUS` or
`ALEXANDRIA_AOZORA_CORPUS`; otherwise the scripts look here.

## aozora/ — Japanese, public domain

Aozora Bunko (https://www.aozora.gr.jp/). Public domain in Japan; these authors
died more than 70 years ago.

| file | work | author |
| --- | --- | --- |
| `kokoro.txt` / `.html` | Kokoro | Natsume Sōseki |
| `ningen.txt` / `.html` | Ningen Shikkaku | Dazai Osamu |
| `rashomon.txt` / `.html` | Rashōmon | Akutagawa Ryūnosuke |

Used to test quote-aware chunking against 「」 and 『』 delimiters, which behave
differently from Western quotation marks.

## chinese/ — Chinese quotation attribution

| file | contents |
| --- | --- |
| `wp_train_instances.json`, `wp_dev_instances.json`, `wp_test.json` | World of Plainness (WP) splits |
| `wp_names.txt` | speaker roster |
| `jy_test.json` | Jin Yong (JY) test split |

From the Chinese speaker-identification datasets released under Apache-2.0 and
recorded in `THIRD_PARTY_NOTICES.md`. These are the corpora whose existence was
wrongly denied on 2026-08-03 before being found — see the note in that file.

## Reconstructing

Place the directories as `ab_test_runtime/corpora/aozora` and
`ab_test_runtime/corpora/chinese`. The experiments fail loudly with a missing
path rather than silently scoring nothing.
