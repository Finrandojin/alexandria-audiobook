"""Resolve the U+FFFD sites deterministic rules refuse to guess at.

WHY A MODEL AND NOT ANOTHER RULE. `repair_source_encoding.py` restores 81.5% of
index18 from context alone and stops, because what is left genuinely cannot be
settled by pattern:

    coup d??tat        an apostrophe and an e-acute
    the dust away?     an ellipsis, or a dash
    the knights? view  a plural possessive, or a closing quote

A rule that guessed here would replace a lost character with a confidently
wrong one, which is worse than leaving it visible. Reading the surrounding
sentence is exactly the judgement a language model can make and a regex cannot.

THE UNIT OF REPAIR IS THE RUN, NOT THE CHARACTER. 318 of the 892 sites are two
to four replacement characters in a row, and they are not necessarily the same
character - "coup d??tat" is an apostrophe followed by an e-acute. Each run is
therefore asked for as a whole and answered with a string of exactly that
length.

WHAT THE MODEL IS NOT ALLOWED TO DO. It returns characters, never text. Every
answer is checked to be the right length and drawn from an allowlist of
characters a broken decode actually destroys - accented Latin letters,
typographic punctuation, currency and symbols. Anything else, and the site is
left as U+FFFD. The model cannot rewrite a word, delete a sentence, or
introduce ASCII that would silently alter spelling, because a wrong repair here
is invisible: nobody re-reads a book to check that an accent came back
correctly.

MEASURED OUTCOME, 2026-08-09: DO NOT APPLY THIS BLIND. Trialled on 40 runs of
index18 against qwen3-14b, roughly half the accepted answers were wrong on
inspection:

    correct:  "English translation (c) 2019"        -> copyright sign
              "blew the dust away..."               -> ellipsis
              "the knights' vision"                 -> possessive apostrophe
    wrong:    "(c)KAZUMA KAMACHI 2009"              -> answered opening quote
              "Haimura, Kiyotaka, 1973- illustrator" -> answered copyright sign
              "giggle, \"Well, what will..."          -> answered apostrophe

An earlier prompt did worse still, defaulting to an apostrophe nearly
everywhere. Rewriting it with an explicit decision order raised the number of
answers ACCEPTED from 10 to 23 without making them more correct - the metric
that moved was not the one that mattered.

The validator cannot catch this. It checks length and character class, which a
wrong-but-plausible quotation mark passes. Only reading the proposals against
their context finds it, and that is a human's job here.

So this script exists to PROPOSE, not to apply. Use --apply only after reading
the report, and prefer it as a review aid over an automatic repair. The
deterministic pass in repair_source_encoding.py remains the trustworthy one:
it restores 81.5% of index18 and refuses the rest by design.

The original file is never modified.
"""
import argparse
import json
import os
import re
import sys
import unicodedata

APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP)

FFFD = "�"
CONTEXT = 90

# Characters a mis-decode plausibly destroyed. Deliberately excludes plain
# ASCII letters and digits: those survive any of these codecs intact, so a
# model proposing one is confused rather than correct, and accepting it would
# change a word's spelling silently.
ALLOWED = set(
    "‘’“”‚„…–—―•·§¶©®™°±×÷¢£¥€«»‹›†‡‰′″"
    "áàâäãåāăąéèêëēĕėęěíìîïĩīĭįıóòôöõōŏőøúùûüũūŭůűųñńņňçćĉċčšśŝşžźżýÿŷ"
    "ÁÀÂÄÃÅĀÉÈÊËĒÍÌÎÏĪÓÒÔÖÕŌØÚÙÛÜŪÑŃÇĆČŠŚŽŹÝ"
    "ßæÆœŒðÐþÞ"
)

SYSTEM = (
    "You repair text where a decoding error replaced characters with the "
    "Unicode replacement character. You answer only with the characters that "
    "belonged there. You never rewrite, translate, or explain the text."
)

TEMPLATE = """Each item below is a passage from an English novel where a decoding error destroyed one or more characters, shown as {marker}. Every destroyed character was NON-ASCII.

Decide from the surrounding words WHICH character it was. Do not default to an apostrophe - that is the most common mistake. Work through these in order:

1. Does a speech turn START here? (the marker sits before a capitalised word that begins dialogue, or follows a line break, or follows "said," / "asked," / a comma+space) -> opening quote {open_q}
2. Does a speech turn END here? (the marker follows . ! ? or , and is followed by a space, a line break, or a dialogue tag like "he said") -> closing quote {close_q}
3. Is it INSIDE a word between two letters? (don_t, author_s, O_Brien) -> apostrophe {apos}
4. Does a sentence TRAIL OFF with nothing following on the line? (away_, said_, now_) -> ellipsis {ellipsis}
5. Is it an interruption or aside between words? -> em dash {emdash}
6. Is a non-English word involved? (coup d_tat, caf_, na_ve) -> the accented letter (e-acute, e-grave, i-diaeresis, and so on)

The run length is given. Your answer for that item must be exactly that many characters - a run of 2 means TWO characters were destroyed there, often different ones.

Return ONLY a JSON array, one object per item:
[{{"n": 0, "chars": "{open_q}"}}, {{"n": 1, "chars": "{apos}\u00e9"}}]

Items:
{items}"""


def find_runs(text):
    runs = []
    for match in re.finditer(FFFD + "+", text):
        runs.append((match.start(), match.end() - match.start()))
    return runs


def build_items(text, runs):
    items = []
    for index, (start, length) in enumerate(runs):
        before = text[max(0, start - CONTEXT):start]
        after = text[start + length:start + length + CONTEXT]
        passage = (before + (FFFD * length) + after).replace("\n", " ⏎ ")
        items.append({"n": index, "run_length": length, "passage": passage,
                      "start": start})
    return items


def validate(answer, length):
    """-> (chars, reason). chars is None when the answer must be refused."""
    if not isinstance(answer, str):
        return None, "not_a_string"
    if len(answer) != length:
        return None, f"wrong_length_{len(answer)}_expected_{length}"
    for char in answer:
        if char not in ALLOWED:
            name = unicodedata.name(char, "UNNAMED")
            return None, f"not_allowed:{name}"
    return answer, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N runs (for a trial)")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    import json as _json
    from openai import OpenAI

    with open(args.source, encoding="utf-8") as handle:
        text = handle.read()

    runs = find_runs(text)
    if args.limit:
        runs = runs[:args.limit]
    items = build_items(text, runs)
    print(f"  {len(runs)} runs to resolve "
          f"({sum(l for _, l in runs)} characters)")

    # Read the same config the generator uses, rather than a second source of
    # truth for the endpoint (Rule 15).
    with open(os.path.join(APP, "config.json"), encoding="utf-8") as handle:
        config = _json.load(handle)
    mode = config.get("llm_mode", "local")
    llm = config.get("llm_remote" if mode == "remote" else "llm_local") \
        or config.get("llm") or {}
    model_name = llm.get("model_name")
    client = OpenAI(base_url=llm.get("base_url"),
                    api_key=llm.get("api_key") or "local")
    print(f"  endpoint {llm.get('base_url')} model {model_name}")

    decisions = {}
    refusals = {}
    for offset in range(0, len(items), args.batch):
        batch = items[offset:offset + args.batch]
        listing = "\n".join(
            f'{item["n"]}. run_length={item["run_length"]}  '
            f'...{item["passage"]}...' for item in batch)
        prompt = TEMPLATE.format(marker=FFFD, items=listing,
                                 open_q="\u201c", close_q="\u201d",
                                 apos="\u2019", ellipsis="\u2026",
                                 emdash="\u2014")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=1200)
            content = response.choices[0].message.content or ""
            start = content.find("[")
            parsed = json.loads(content[start:content.rfind("]") + 1])
        except Exception as exc:                            # noqa: BLE001
            print(f"    batch {offset}: failed ({exc}); left unrepaired")
            continue
        by_n = {item["n"]: item for item in batch}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            item = by_n.get(entry.get("n"))
            if item is None:
                continue
            chars, reason = validate(entry.get("chars"), item["run_length"])
            if chars is None:
                refusals[item["n"]] = reason
            else:
                decisions[item["n"]] = chars
        print(f"    {offset + len(batch)}/{len(items)} "
              f"resolved={len(decisions)} refused={len(refusals)}", flush=True)

    # Rebuild right-to-left so earlier offsets stay valid.
    repaired = text
    for index in sorted(decisions, key=lambda i: -items[i]["start"]):
        item = items[index]
        start = item["start"]
        repaired = (repaired[:start] + decisions[index]
                    + repaired[start + item["run_length"]:])

    stem, extension = os.path.splitext(args.source)
    out_path = args.out or f"{stem}.llm_repaired{extension}"
    report_path = args.report or f"{stem}.llm_repair_report.json"

    report = {
        "source": os.path.abspath(args.source),
        "runs_total": len(runs),
        "runs_resolved": len(decisions),
        "runs_refused": len(refusals),
        "runs_unanswered": len(runs) - len(decisions) - len(refusals),
        "replacement_chars_before": text.count(FFFD),
        "replacement_chars_after": repaired.count(FFFD),
        "refusal_reasons": {},
        "samples": [],
        "applied": bool(args.apply),
    }
    for reason in refusals.values():
        key = reason.split(":")[0]
        report["refusal_reasons"][key] = report["refusal_reasons"].get(key, 0) + 1
    for index in list(decisions)[:40]:
        item = items[index]
        report["samples"].append({
            "chars": decisions[index],
            "passage": item["passage"][CONTEXT - 45:CONTEXT + 45],
        })

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    if args.apply:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(repaired)

    print(f"\n  resolved {len(decisions)} / {len(runs)} runs")
    print(f"  U+FFFD {text.count(FFFD)} -> {repaired.count(FFFD)}")
    print(f"  refusals: {report['refusal_reasons']}")
    print(f"  report: {report_path}")
    print(f"  file: {out_path if args.apply else '(dry run)'}")


if __name__ == "__main__":
    main()
