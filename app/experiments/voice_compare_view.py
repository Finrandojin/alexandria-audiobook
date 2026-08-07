"""Look at the generated voice next to the human reading the same line.

WHY THIS EXISTS. `ljspeech_score.py` measures four things - speaker embedding,
pitch contour, spectral envelope, duration - and nobody has ever LOOKED at the
audio those numbers describe. 450 generated clips exist, paired line-for-line
with their human originals, and not one spectrogram has been plotted.

That gap has already cost something. The Chinese arm scored `human_vs_human`
0.691 while its own arms reached 0.720 and 0.765 - a narrator matching herself
worse than a synthetic voice matched her. It took writing an anchor-validity
check to notice. A picture of those clips would have shown it immediately.

WHAT THIS IS NOT. Not a metric, and deliberately not a substitute for one.
Comparing raw waveforms sample-by-sample is meaningless for TTS: two identical
sounding readings differ completely in phase and micro-timing. Eyes are for
catching the gross failure a number hides - a clip that is silence, a voice an
octave out, a boundary in the wrong place - not for ranking arms.

WHAT IS DRAWN, per line and per arm:

    waveform        gross shape, silence, clipping, truncation
    mel-spectrogram timbre and formant structure, the thing MCD summarises
    f0 contour      the prosody, overlaid on the human so the shape can be
                    compared directly rather than through one correlation number

Self-contained HTML with the images and audio inlined, so it can be opened
anywhere without the repo.
"""
import argparse
import base64
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(REPO, "app")
sys.path.insert(0, APP)


def load_audio(path, sr=22050):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def envelope(y, sr, columns=900):
    """Per-column (min, max) amplitude — the filled shape a DAW draws.

    Plotting raw samples gives a thin scribble that hides the thing you want to
    see. Peak-per-column is what Audacity and every editor shows, and it is
    what makes a missing syllable or a truncated tail visible at a glance.
    """
    import numpy as np
    if len(y) == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    step = max(len(y) // columns, 1)
    usable = (len(y) // step) * step
    block = y[:usable].reshape(-1, step)
    lo, hi = block.min(axis=1), block.max(axis=1)
    t = np.linspace(0, len(y) / sr, len(lo))
    return t, lo, hi


def rms_envelope(y, sr, columns=900):
    """Loudness contour. This is what can honestly be SUBTRACTED between two
    takes: raw samples cannot, because two identical-sounding readings differ
    completely in phase, and their difference is nearly the sum of both. The
    energy envelope survives that and still carries pacing and emphasis.
    """
    import numpy as np
    if len(y) == 0:
        return np.zeros(1)
    step = max(len(y) // columns, 1)
    usable = (len(y) // step) * step
    block = y[:usable].reshape(-1, step)
    return np.sqrt((block ** 2).mean(axis=1))


def resample_curve(curve, n):
    """Stretch a curve onto n points, so two takes of different length can be
    compared by SHAPE. Timing difference is not thrown away - it is reported
    separately as dur_ratio, and shown untouched in the overlay above."""
    import numpy as np
    if len(curve) < 2:
        return np.zeros(n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(curve)), curve)


def voice_quality(path):
    """Jitter, shimmer and HNR via Praat's own algorithms (parselmouth).

    WHY THESE AND NOT MORE OF OURS. A study comparing speakers found jitter the
    single most discriminative measure, then F2, then shimmer, then HNR - and
    duration among the LEAST. Our metric set was ECAPA, f0 contour, MCD and
    duration, so it measured none of the top four and one of the bottom.

    What they catch that nothing else here does: a synthetic voice can be too
    CLEAN. Human phonation wobbles cycle to cycle; a vocoder often does not.
    Near-zero jitter against a human's natural variation is a tell that no
    embedding distance reports.

    Returns None values rather than zeros when there is too little voicing -
    which is itself informative on short clips.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {}
    try:
        snd = parselmouth.Sound(path)
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 350)
        out = {
            "jitter_local": call(pp, "Get jitter (local)",
                                 0, 0, 0.0001, 0.02, 1.3),
            "shimmer_local": call([snd, pp], "Get shimmer (local)",
                                  0, 0, 0.0001, 0.02, 1.3, 1.6),
            "hnr_db": call(call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0),
                           "Get mean", 0, 0),
        }
        return {k: (None if v is None or v != v else round(float(v), 4))
                for k, v in out.items()}
    except Exception:                                       # noqa: BLE001
        return {}


def formant_tracks(path, max_formant=5500):
    """F1..F3 over time. F2 is a top-two speaker discriminator and nothing in
    the existing metric set looks at it."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None
    import numpy as np
    snd = parselmouth.Sound(path)
    fo = snd.to_formant_burg(max_number_of_formants=5,
                             maximum_formant=max_formant)
    times = np.arange(fo.get_start_time(), fo.get_end_time(), 0.01)
    tracks = {}
    for n in (1, 2, 3):
        vals = []
        for t in times:
            v = call(fo, "Get value at time", n, float(t), "Hertz", "Linear")
            vals.append(np.nan if v is None or v != v else v)
        tracks[f"F{n}"] = np.array(vals)
    return times, tracks


def vocal_tract_length(path, c=35000.0):
    """Estimated vocal tract length in cm, from formant dispersion.

    WHY THIS AND NOT A GENDER LABEL. This project deliberately refuses to
    classify gender from pitch - `test_pitch_is_not_used_as_a_gender_classifier`
    asserts that a 90 Hz voice is "unknown", and that a description of "warm
    mezzo" beats the pitch. That decision is right: male and female f0
    distributions overlap heavily and a naive classifier misgenders low-voiced
    women and high-voiced men.

    This measures something different and safer. Vocal tract length is the
    physical property that drives most of what listeners hear as vocal size,
    and it is estimated from the SPACING of formants rather than from pitch. It
    is used here only to ask whether an arm PRESERVES the speaker's tract
    length - a comparison between two clips of the same person - never to
    assign anyone a category.

    Fitch's formant dispersion: Df = mean(F(i+1) - F(i)), VTL = c / (2 * Df).
    """
    import numpy as np
    got = formant_tracks(path)
    if not got:
        return None
    _t, tracks = got
    means = []
    for n in ("F1", "F2", "F3"):
        v = tracks[n]
        v = v[~np.isnan(v)]
        if len(v) < 5:
            return None
        means.append(float(np.median(v)))
    disp = np.mean(np.diff(means))
    if disp <= 0:
        return None
    return round(c / (2.0 * disp), 2)


def pitch_stats(path, sr=22050):
    """Median and 10-90 percentile spread of f0, in Hz. Reported as
    PRESERVATION against the human, not as an identity claim."""
    import numpy as np
    y, _ = load_audio(path, sr)
    _t, f = f0_contour(y, sr)
    if f is None:
        return {}
    v = f[~np.isnan(f)]
    if len(v) < 5:
        return {}
    return {"f0_median": round(float(np.median(v)), 1),
            "f0_spread": round(float(np.percentile(v, 90)
                                     - np.percentile(v, 10)), 1)}


def alignment_path(human_path, arm_path, sr=22050):
    """DTW path between the two takes, as (human_time, arm_time) pairs.

    The convention TTS papers use: ground truth on x, generated on y, perfect
    synchronisation on the diagonal. Deviation from the diagonal IS the timing
    error, readable directly - which the time-normalised energy difference
    cannot show, because normalising is exactly what throws timing away.
    """
    import librosa
    import numpy as np
    from librosa.sequence import dtw
    a, _ = librosa.load(human_path, sr=sr, mono=True)
    b, _ = librosa.load(arm_path, sr=sr, mono=True)
    hop = 256
    ma = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=20, hop_length=hop)
    mb = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=20, hop_length=hop)
    _cost, path = dtw(ma, mb, subseq=False)
    path = path[::-1]
    sec = hop / sr
    return path[:, 0] * sec, path[:, 1] * sec


def f0_contour(y, sr):
    import librosa
    import numpy as np
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=350, sr=sr)
    if f0 is None:
        return None, None
    t = librosa.times_like(f0, sr=sr)
    return t, np.where(voiced, f0, np.nan)


def render_line(row, arms, out_dir, sr=22050, wide_ms=5, narrow_ms=25):
    """One figure per line: waveform + mel + f0, human against each arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import numpy as np

    if "human" not in arms or not os.path.exists(arms["human"]):
        return None
    gen = [(n, p) for n, p in arms.items()
           if n != "human" and p and os.path.exists(p)]
    if not gen:
        return None

    human_y, _ = load_audio(arms["human"], sr)
    human_dur = len(human_y) / sr
    ht, hlo, hhi = envelope(human_y, sr)
    hrms = rms_envelope(human_y, sr)

    # Measured once for the human, then compared against each arm.
    h_vtl = vocal_tract_length(arms["human"])
    h_pitch = pitch_stats(arms["human"], sr)
    h_vq = voice_quality(arms["human"])

    ncol = len(gen)
    fig, axes = plt.subplots(8, ncol, figsize=(6.2 * ncol, 21.5), squeeze=False)

    for col, (name, path) in enumerate(gen):
        y, _ = load_audio(path, sr)
        dur = len(y) / sr
        t, lo, hi = envelope(y, sr)

        # 1 — OVERLAY ON REAL TIME. Both takes on the same seconds axis, so
        # where they stop lining up is the picture, not a number. The human is
        # the grey bed; the arm is drawn over it.
        ax = axes[0][col]
        ax.fill_between(ht, hlo, hhi, color="#8a8a95", alpha=0.55, lw=0,
                        label=f"human {human_dur:.2f}s")
        ax.fill_between(t, lo, hi, color="#1f5fd0", alpha=0.55, lw=0,
                        label=f"{name} {dur:.2f}s")
        ax.axvline(human_dur, color="#8a8a95", ls=":", lw=1)
        ax.set_xlim(0, max(human_dur, dur) * 1.02)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{name} vs human  —  {dur/human_dur:.2f}x duration",
                     fontsize=11)
        if col == 0:
            ax.set_ylabel("waveform overlay")

        # 2 — DIFFERENCE, on the loudness envelope, time-normalised.
        # Raw samples are not subtractable between takes (phase), and the two
        # takes differ in length anyway, so both curves are stretched onto a
        # common axis and the ENERGY is differenced. Above zero the arm is
        # louder than the human at that point in the line, below it quieter.
        ax = axes[1][col]
        n = 900
        a = resample_curve(rms_envelope(y, sr), n)
        b = resample_curve(hrms, n)
        d = a - b
        x = np.linspace(0, 1, n)
        ax.fill_between(x, 0, np.clip(d, 0, None), color="#1f9d55", alpha=.75,
                        lw=0, label="louder than human")
        ax.fill_between(x, np.clip(d, None, 0), 0, color="#c1121f", alpha=.75,
                        lw=0, label="quieter")
        ax.axhline(0, color="#333", lw=.8)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("position through the line (time-normalised)")
        if col == 0:
            ax.set_ylabel("energy  arm − human")

        # 3 — ALIGNMENT DIAGONAL, the convention TTS papers use: ground truth
        # on x, generated on y, perfect synchronisation on the diagonal.
        # Deviation from the diagonal IS the timing error. The energy panel
        # above cannot show this, because time-normalising is precisely what
        # discards timing.
        ax = axes[2][col]
        try:
            hx, ay = alignment_path(arms["human"], path, sr)
            ax.plot(hx, ay, lw=1.6, color="#1f5fd0")
            lim = max(human_dur, dur)
            ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#8a8a95",
                    label="perfect sync")
            drift = float(np.max(np.abs(ay - hx))) if len(hx) else 0.0
            ax.set_title(f"worst drift {drift*1000:.0f} ms", fontsize=9)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.legend(fontsize=8, loc="lower right")
        except Exception as exc:                            # noqa: BLE001
            ax.text(.5, .5, f"alignment failed\n{str(exc)[:40]}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel("human (s)")
        if col == 0:
            ax.set_ylabel("arm (s)")

        # 4 and 5 — WIDEBAND spectrograms, human directly above the arm.
        # Window length is the whole point: ~5ms (wideband) resolves FORMANTS,
        # ~25ms (narrowband) resolves HARMONICS. The default mel setting shows
        # neither cleanly, which is what the earlier version of this plot did.
        def spec_panel(ax, sig, label, win_ms):
            n_fft = 1 << int(np.ceil(np.log2(sr * win_ms / 1000)))
            S = np.abs(librosa.stft(sig, n_fft=n_fft,
                                    hop_length=max(n_fft // 8, 1)))
            librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                     sr=sr, hop_length=max(n_fft // 8, 1),
                                     x_axis="time", y_axis="linear", ax=ax,
                                     cmap="magma")
            ax.set_ylim(0, 5000)
            ax.set_xlabel("")
            ax.set_ylabel(label if col == 0 else "")
            ax.text(0.01, 0.93, label, transform=ax.transAxes, fontsize=9,
                    color="w", va="top")

        spec_panel(axes[3][col], human_y, "wideband — human", wide_ms)
        spec_panel(axes[4][col], y, f"wideband — {name}", wide_ms)

        # 6 — FORMANTS. F2 is a top-two speaker discriminator per the voice
        # quality literature, and nothing in the existing metric set looks at
        # it. Human dashed behind, arm solid.
        ax = axes[5][col]
        try:
            ht_f, htr = formant_tracks(arms["human"])
            at_f, atr = formant_tracks(path)
            for n, colr in (("F1", "#c1121f"), ("F2", "#1f9d55"),
                            ("F3", "#7048c4")):
                ax.plot(ht_f, htr[n], ls="--", lw=1.1, alpha=.55, color=colr)
                ax.plot(at_f, atr[n], lw=1.4, color=colr, label=n)
            ax.legend(fontsize=8, loc="upper right", ncol=3)
            ax.set_ylim(0, 4000)
        except Exception as exc:                            # noqa: BLE001
            ax.text(.5, .5, f"formants failed\n{str(exc)[:40]}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlim(0, max(human_dur, dur) * 1.02)
        if col == 0:
            ax.set_ylabel("formants (Hz)\ndashed = human")

        # 7 — PRESERVATION. Every bar is the arm as a fraction of the human, so
        # 1.0 is "kept it" and the distance from the centre line is the defect.
        # None of this labels the speaker; it asks whether the arm carried the
        # speaker's own properties across. That is the distinction that lets a
        # vocal-tract measure exist in a project that refuses pitch-based
        # gender classification.
        ax = axes[6][col]
        a_vtl = vocal_tract_length(path)
        a_pitch = pitch_stats(path, sr)
        a_vq = voice_quality(path)
        labels, ratios, notes = [], [], []

        def add(label, h, a, invert=False):
            if h in (None, 0) or a is None:
                labels.append(label); ratios.append(np.nan); notes.append("n/a")
                return
            r = a / h
            labels.append(label); ratios.append(r)
            notes.append(f"{a:.4g} vs {h:.4g}")

        add("vocal tract len", h_vtl, a_vtl)
        add("f0 median", h_pitch.get("f0_median"), a_pitch.get("f0_median"))
        add("f0 spread", h_pitch.get("f0_spread"), a_pitch.get("f0_spread"))
        add("jitter", h_vq.get("jitter_local"), a_vq.get("jitter_local"))
        add("shimmer", h_vq.get("shimmer_local"), a_vq.get("shimmer_local"))
        add("HNR", h_vq.get("hnr_db"), a_vq.get("hnr_db"))

        ypos = np.arange(len(labels))
        vals = np.array([0 if v != v else v - 1.0 for v in ratios])
        colours = ["#1f9d55" if abs(v) <= 0.10 else
                   "#e08a00" if abs(v) <= 0.25 else "#c1121f" for v in vals]
        ax.barh(ypos, vals, color=colours, height=.6)
        ax.axvline(0, color="#333", lw=1)
        for lim in (-0.10, 0.10):
            ax.axvline(lim, color="#8a8a95", ls=":", lw=.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-0.7, 0.7)
        ax.set_xlabel("arm ÷ human − 1   (0 = preserved, dotted = ±10%)")
        for i, (v, n) in enumerate(zip(vals, notes)):
            ax.text(0.66 if v < 0 else -0.66, i, n, fontsize=7, va="center",
                    ha="right" if v < 0 else "left", color="#555")
        if col == 0:
            ax.set_ylabel("preservation")

        # 8 — f0 with the human behind, so prosody is compared to the target
        ax = axes[7][col]
        th, fh = f0_contour(human_y, sr)
        if th is not None:
            ax.plot(th, fh, linewidth=2.0, alpha=0.4, color="#8a8a95",
                    label="human")
        tt, ff = f0_contour(y, sr)
        if tt is not None:
            ax.plot(tt, ff, linewidth=1.5, color="#1f5fd0", label=name)
        ax.set_ylim(50, 360)
        ax.set_xlim(0, max(human_dur, dur) * 1.02)
        ax.legend(fontsize=8, loc="upper right")
        if col == 0:
            ax.set_ylabel("f0 (Hz)")
        ax.set_xlabel("seconds")

    fig.suptitle(f'{row["id"]}   "{str(row.get("text"))[:90]}"', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=88)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def audio_tag(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b = base64.b64encode(fh.read()).decode("ascii")
    return (f'<audio controls preload="none" '
            f'src="data:audio/wav;base64,{b}"></audio>')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generated", default=os.path.join(
        REPO, "ab_test_runtime", "experiments", "ljspeech_generate.json"))
    ap.add_argument("--score", default=None,
                    help="matching *_score.json, to print each line's metrics")
    ap.add_argument("--lines", type=int, default=6)
    ap.add_argument("--pick", default="spread",
                    choices=("spread", "best", "worst", "first"),
                    help="spread: best, median and worst by ecapa, so the view "
                         "is not quietly a highlight reel")
    ap.add_argument("--wide-ms", type=float, default=5,
                    help="wideband window (ms). ~5 resolves formants")
    ap.add_argument("--narrow-ms", type=float, default=25,
                    help="narrowband window (ms). ~25 resolves harmonics")
    ap.add_argument("--out", default=os.path.join(
        REPO, "ab_test_runtime", "voice_compare", "index.html"))
    args = ap.parse_args()

    doc = json.load(open(args.generated, encoding="utf-8"))
    rows = doc.get("rows") or []
    if not rows:
        sys.exit("no rows in the generated artifact")

    scores = {}
    if args.score and os.path.exists(args.score):
        sdoc = json.load(open(args.score, encoding="utf-8"))
        scores = {r["id"]: r for r in (sdoc.get("rows") or [])}

    # SELECTION IS NOT INNOCENT. Showing the first N, or the best N, produces a
    # view that agrees with whatever was hoped for. Spread shows the worst too.
    if scores and args.pick != "first":
        def key(r):
            s = scores.get(r["id"], {})
            arm = s.get("clone") or s.get("lora") or {}
            return arm.get("ecapa") if arm.get("ecapa") is not None else -1
        ranked = sorted([r for r in rows if r["id"] in scores], key=key)
        if args.pick == "worst":
            chosen = ranked[:args.lines]
        elif args.pick == "best":
            chosen = ranked[-args.lines:]
        else:
            n = args.lines
            idx = [int(i * (len(ranked) - 1) / max(n - 1, 1)) for i in range(n)]
            chosen = [ranked[i] for i in sorted(set(idx))]
    else:
        chosen = rows[:args.lines]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    parts = []
    for row in chosen:
        arms = {"human": os.path.join(REPO, row["human_wav"])}
        for k in ("clone_wav", "lora_wav"):
            if row.get(k):
                arms[k.replace("_wav", "")] = os.path.join(REPO, row[k])
        png = render_line(row, arms, os.path.dirname(args.out),
                          wide_ms=args.wide_ms,
                          narrow_ms=args.narrow_ms)
        if not png:
            continue
        s = scores.get(row["id"], {})
        bits = []
        for arm in ("clone", "lora", "human_vs_human"):
            a = s.get(arm) or {}
            if a.get("ecapa") is None:
                continue
            # A metric can be present-and-None: pyin finds too little voicing on
            # short or breathy clips and f0_corr comes back null. Formatting
            # that as a float crashed on the Chinese set, whose clips are the
            # shortest at 3.17s median - exactly where voicing is thinnest.
            def num(key, fmt):
                v = a.get(key)
                return format(v, fmt) if isinstance(v, (int, float)) else "--"
            bits.append(f"{arm} ecapa {num('ecapa', '.3f')} "
                        f"f0 {num('f0_corr', '.2f')} "
                        f"dur {num('dur_ratio', '.2f')}")
        # Voice quality, measured here rather than read from the score
        # artifact, because ljspeech_score has never computed it. Jitter is the
        # most discriminative single measure in the literature and this project
        # had no number for it at all.
        vq_bits = []
        for label, p in (("human", arms.get("human")),
                         *[(n, p) for n, p in arms.items() if n != "human"]):
            q = voice_quality(p) if p and os.path.exists(p) else {}
            if q.get("jitter_local") is not None:
                vq_bits.append(
                    f"{label} jitter {q['jitter_local']*100:.2f}% "
                    f"shimmer {q['shimmer_local']*100:.2f}% "
                    f"HNR {q['hnr_db']:.1f}dB")
        players = "".join(
            f'<div class="p"><span>{n}</span>{audio_tag(p)}</div>'
            for n, p in arms.items())
        parts.append(
            f'<section><h2>{row["id"]}</h2>'
            f'<p class="t">{str(row.get("text"))[:200]}</p>'
            f'<p class="m">{" &nbsp;|&nbsp; ".join(bits)}</p>'
            f'<p class="q">{" &nbsp;|&nbsp; ".join(vq_bits)}</p>'
            f'<img src="data:image/png;base64,{png}" alt="{row["id"]}">'
            f'<div class="players">{players}</div></section>')
        print(f"  rendered {row['id']}")

    html = f"""<meta charset="utf-8"><title>Voice comparison</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;
   padding:0 1rem;background:#fbfbfc;color:#1a1a1c}}
 h1{{font-size:1.5rem;margin-bottom:.2rem}}
 .lede{{color:#555;margin-top:0}}
 section{{background:#fff;border:1px solid #e4e4e8;border-radius:8px;
   padding:1rem 1.2rem;margin:1.4rem 0}}
 h2{{font-size:1rem;margin:0 0 .2rem;font-family:ui-monospace,monospace}}
 .t{{margin:.2rem 0;color:#333}}
 .m{{margin:.2rem 0 .1rem;font-family:ui-monospace,monospace;font-size:.82rem;
   color:#666}}
 .q{{margin:0 0 .8rem;font-family:ui-monospace,monospace;font-size:.82rem;
   color:#8a5a00}}
 img{{width:100%;height:auto;border-radius:4px}}
 .players{{display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:.7rem}}
 .p{{display:flex;align-items:center;gap:.4rem}}
 .p span{{font-family:ui-monospace,monospace;font-size:.8rem;color:#666}}
 audio{{height:32px}}
</style>
<h1>Generated voice vs the human reading the same line</h1>
<p class="lede">Top row waveform, middle mel-spectrogram, bottom f0 contour with
the human drawn behind each arm. Lines chosen by <b>{args.pick}</b> — a spread
across the ECAPA range, so this is not a highlight reel. Eyes here are for the
gross failure a number hides, not for ranking arms.</p>
{"".join(parts)}"""
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)//1024} KB, "
          f"{len(parts)} lines)")
    if not parts:
        sys.exit(3)


if __name__ == "__main__":
    main()
