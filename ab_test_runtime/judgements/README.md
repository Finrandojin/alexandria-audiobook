Judge passes, one folder per book per pass.

    <book>/            gemini, 4-segment context   (superseded)
    <book>_wide/       gemini, 12-segment context  (current)
    <book>_openai/     openai, 12-segment context  (independent second read)
    review/            the human review's exported decisions

Replies go in as `reply_*.json` and may be a bare array, a fenced array, or an
object with a `rows` list; `speaker` and `ANSWER` are both accepted as the
label key. Ids must belong to the book's bundle - a reply covering a different
sample belongs in its own folder, not here, or it blocks the whole book (see
mushoku16_other_sample).

Compare two passes with:

    cd app && ./env/bin/python experiments/judge_agreement.py

which writes the disagreement queue to judgements/disagreements.json.
