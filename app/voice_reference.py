"""Choose the reference clip a LoRA dataset will be anchored to.

WHY THIS EXISTS. `train_lora.py` extracts the speaker embedding from ONE
reference clip and uses it for every training sample, per the official Qwen3-TTS
fine-tuning approach. The dataset builder chose that clip as `ref_index = 0` -
whatever sample happened to be first - and nothing checked it against the rest
of the dataset.

Measured across the 75 shipped adapters on 2026-08-07:

    correlation(reference matches its dataset, adapter quality) = +0.76
    reference mismatched (<0.3):  7 adapters, 6 of them poor  (86%)
    reference matching:          67 adapters, 9 of them poor  (13%)

A mismatched reference makes an adapter 6.4x more likely to fail. The worst
case, `husky_baritone_20s_m_anime`, was anchored to a clip scoring **-0.026**
against its own dataset - actively not that speaker - and produced an adapter
scoring 0.004. A representative clip scoring 0.882 was sitting in the same data.

THE MEDOID IS THE FIX. The medoid is the clip most similar to all the others,
so it is representative by construction and robust to a minority of bad clips -
which is the failure mode, since a dataset with a few misdiarized clips still
has a clear majority speaker. Across 74 datasets the medoid beat the existing
reference by a median of +0.07, and by more than 0.15 on 14 of them.

DEGRADES RATHER THAN BREAKS. The speaker model lives in the sibling
interpreter, not in `app/env`. When it is unavailable this returns None and the
caller keeps its existing behaviour, because making dataset creation
hard-depend on a second environment would be a worse failure than an
occasionally poor reference. The caller logs which path was taken.
"""
import json
import os
import statistics
import subprocess

APP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(APP)

SIBLING_PY = os.environ.get(
    "ALEXANDRIA_SIBLING_PYTHON",
    os.path.join(os.path.dirname(REPO), "alexandria-audiobook.git",
                 "app", "env", "bin", "python"))

# Bounded on purpose: this runs inside a save request. 12 clips is 66 pairwise
# comparisons, enough to identify the majority speaker, and the cost grows
# quadratically.
MAX_CLIPS = 12


def _speaker_similarities(pairs, timeout=600):
    """Cosine similarity per pair, or None if the model is unavailable."""
    if not pairs or not os.path.exists(SIBLING_PY):
        return None
    script = os.path.join(APP, "experiments", "_ecapa_batch.py")
    if not os.path.exists(script):
        return None
    try:
        out = subprocess.run(
            [SIBLING_PY, script],
            input=json.dumps([[os.path.abspath(a), os.path.abspath(b)]
                              for a, b in pairs]),
            capture_output=True, text=True, timeout=timeout, cwd=APP)
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def select_reference_sample(wav_paths, max_clips=MAX_CLIPS):
    """-> (index into wav_paths, median similarity) or (None, None).

    None means "could not decide" - the model was unavailable, or too few
    clips to compare - and the caller should keep whatever it was going to do.
    It never guesses: returning index 0 on failure would be indistinguishable
    from a real decision and would hide exactly the defect this exists for.
    """
    usable = [(i, p) for i, p in enumerate(wav_paths)
              if p and os.path.exists(p)]
    if len(usable) < 3:
        return None, None
    sample = usable[:max_clips]
    pairs, index = [], []
    for a in range(len(sample)):
        for b in range(a + 1, len(sample)):
            pairs.append((sample[a][1], sample[b][1]))
            index.append((a, b))
    sims = _speaker_similarities(pairs)
    if not sims or len(sims) != len(pairs):
        return None, None
    scores = {a: [] for a in range(len(sample))}
    for (a, b), value in zip(index, sims):
        if value is None:
            continue
        scores[a].append(value)
        scores[b].append(value)
    medians = {a: statistics.median(v) for a, v in scores.items() if v}
    if not medians:
        return None, None
    best = max(medians, key=medians.get)
    return sample[best][0], round(medians[best], 4)
