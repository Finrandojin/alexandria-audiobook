"""Which stage actually loses the most lines?

I asserted repeatedly through this investigation that segmentation was "the
largest uninstrumented error source" and steered work toward it on that basis.
The two stages were never put on the same denominator. They are here.

  segmentation misfiling   a judged row whose true label is NOT_DIALOGUE: the
                           segmenter sent narration to TTS as character speech
  attribution error        a correctly-segmented spoken line given the wrong
                           speaker, in the shippable stack with the adapter on

Both are counted per line delivered to TTS, over the same four books.
"""
import glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LEDGER = REPO + "/ab_test_runtime/experiments"


def main():
    seg_n = seg_bad = 0
    for p in sorted(glob.glob(REPO + "/ab_test_runtime/fixtures_draft/labelling_bundle__*.json")):
        for e in json.load(open(p))["entries"]:
            seg_n += 1
            seg_bad += e.get("expected_speaker") == "NOT_DIALOGUE"

    att_n = att_bad = 0
    for p in sorted(glob.glob(LEDGER + "/lora_serving_eval__local-rocm*.json")):
        for r in json.load(open(p))["rows"]:
            if r["arm"] != "lora":
                continue
            att_n += 1
            att_bad += not r.get("correct")

    if not (seg_n and att_n):
        print("need both a labelling bundle and a shippable-stack artifact")
        return
    s, a = seg_bad / seg_n, att_bad / att_n
    print("Per line delivered to TTS, same pipeline, same four books:\n")
    print(f"  segmentation misfiling  {seg_bad:4}/{seg_n:<5} = {s*100:5.2f}%")
    print(f"  attribution error       {att_bad:4}/{att_n:<5} = {a*100:5.2f}%")
    print(f"\n  attribution is {a/s:.1f}x larger")
    print("\n  I called segmentation 'the largest uninstrumented error source'")
    print("  repeatedly. On a common denominator it is six times SMALLER than")
    print("  attribution error. What was true is a different sentence:")
    print("  segmentation is the least MEASURED stage - attribution has a gold")
    print("  standard, a ledger and twenty harnesses, segmentation has 46")
    print("  labels. Biggest problem and least-measured problem are not the")
    print("  same thing, and conflating them sent effort the wrong way.")


if __name__ == "__main__":
    main()
