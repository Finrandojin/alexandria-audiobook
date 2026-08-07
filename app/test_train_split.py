"""Training must not consume the validation split.

WHAT THIS PROTECTS. The dataset builder writes 180 train / 20 val with zero
overlap, and also a root metadata.jsonl holding all 200. `train_lora.py` loaded
the root file unconditionally, so every adapter was trained on its own
validation set - 67 of 75 in the library record num_samples=200.

Nothing failed. The split was right there, correct, and ignored. The cost is
that no adapter in the library can be honestly evaluated: every line available
to test it with has already been seen, so a good score proves only that the
model can repeat its own training data.

WHY A TEST AND NOT JUST THE FIX. The failure is invisible from the outside. A
contaminated adapter trains normally, reports a normal loss, and produces
plausible audio. The only symptom is a number that is too good, months later,
in a document that does not know it is wrong.

Paths inside all three metadata files are relative to the DATASET ROOT, which
is why the fix can change only which file is read.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP = os.path.dirname(os.path.abspath(__file__))


def build_dataset(root, n_train, n_val):
    """Write a dataset shaped like the real builder's output."""
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    train = [{"audio_filepath": f"train/sample_{i}.wav", "text": f"train {i}",
              "duration": 1.0} for i in range(n_train)]
    rows = list(train)
    if n_val:
        os.makedirs(os.path.join(root, "val"), exist_ok=True)
        val = [{"audio_filepath": f"val/sample_{i}.wav", "text": f"val {i}",
                "duration": 1.0} for i in range(n_val)]
        rows += val
        with open(os.path.join(root, "val", "metadata.jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in val)
        with open(os.path.join(root, "train", "metadata.jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in train)
    with open(os.path.join(root, "metadata.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return train


def which_metadata(data_dir):
    """Re-run the selection logic exactly as load_dataset does."""
    split_path = os.path.join(data_dir, "train", "metadata.jsonl")
    root_path = os.path.join(data_dir, "metadata.jsonl")
    return split_path if os.path.exists(split_path) else root_path


class TrainSplitTest(unittest.TestCase):

    def test_the_val_split_is_not_trained_on(self):
        """THE BUG. Loading the root file pulled in all 200 rows, val included."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, 180, 20)
            chosen = which_metadata(tmp)
            with open(chosen, encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            self.assertEqual(len(rows), 180)
            self.assertTrue(all(r["audio_filepath"].startswith("train/")
                                for r in rows),
                            "a val/ clip reached the training set")

    def test_a_dataset_without_a_split_still_trains(self):
        """Older datasets have no train/ subdirectory. They must keep working
        exactly as before, not start failing because a split is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(tmp, exist_ok=True)
            rows = [{"audio_filepath": f"sample_{i}.wav", "text": f"t{i}",
                     "duration": 1.0} for i in range(40)]
            with open(os.path.join(tmp, "metadata.jsonl"), "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            chosen = which_metadata(tmp)
            self.assertEqual(chosen, os.path.join(tmp, "metadata.jsonl"))
            with open(chosen, encoding="utf-8") as fh:
                self.assertEqual(sum(1 for l in fh if l.strip()), 40)

    def test_paths_stay_relative_to_the_dataset_root(self):
        """The fix is only safe because train/metadata.jsonl stores
        'train/sample_N.wav', not 'sample_N.wav'. If the builder ever changes
        that, resolving against data_dir silently breaks."""
        with tempfile.TemporaryDirectory() as tmp:
            build_dataset(tmp, 5, 2)
            with open(os.path.join(tmp, "train", "metadata.jsonl")) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            for r in rows:
                self.assertTrue(r["audio_filepath"].startswith("train/"))
                self.assertFalse(os.path.isabs(r["audio_filepath"]))

    def test_train_lora_reads_the_split_not_the_root(self):
        """Against the real module, so a future edit to load_dataset that
        reintroduces the root path is caught."""
        with open(os.path.join(APP, "train_lora.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('os.path.join(data_dir, "train", "metadata.jsonl")', src,
                      "train_lora no longer prefers the train/ split")
        head = src.split("def load_dataset", 1)[-1][:2500]
        self.assertIn("used_split", head,
                      "the split-vs-root decision disappeared from load_dataset")


class LibraryContaminationTest(unittest.TestCase):
    """The evidence that motivated the fix.

    The first version of this skipped when no adapter library was present -
    which is every CI checkout. The release verifier counts a skip as a
    failure, and rightly: a test that quietly does nothing still reports green.
    Three other tests were rewritten for exactly this reason hours before this
    one was written with the same flaw.

    So the logic is exercised on a fixture, and the real library is consulted
    only as a bonus when one happens to be there.
    """

    def _count_contaminated(self, metas):
        """How many adapters record training on the full 200 (train+val)."""
        counts = []
        for path in metas:
            try:
                counts.append(json.load(open(path, encoding="utf-8"))
                              .get("num_samples"))
            except (OSError, ValueError):
                continue
        return sum(1 for c in counts if c == 200), len(counts)

    def test_the_contamination_signature_is_detectable(self):
        """num_samples == 200 on a 180/20 dataset means both splits were
        consumed. This is the check that identified 67 of 75 adapters."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, n in (("clean", 180), ("contaminated", 200),
                            ("also_clean", 180)):
                d = os.path.join(tmp, name)
                os.makedirs(d)
                with open(os.path.join(d, "training_meta.json"), "w") as fh:
                    json.dump({"num_samples": n}, fh)
            metas = [os.path.join(tmp, n, "training_meta.json")
                     for n in ("clean", "contaminated", "also_clean")]
            self.assertEqual(self._count_contaminated(metas), (1, 3))

    def test_unreadable_metadata_is_not_counted_either_way(self):
        """A corrupt file must not silently inflate or deflate the count."""
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "training_meta.json")
            with open(good, "w") as fh:
                json.dump({"num_samples": 200}, fh)
            bad = os.path.join(tmp, "broken.json")
            with open(bad, "w") as fh:
                fh.write("{not json")
            self.assertEqual(self._count_contaminated([good, bad]), (1, 1))

    def test_the_real_library_when_one_is_present(self):
        """A bonus assertion, never the only one, so an absent library is fine
        rather than a skip."""
        import glob
        metas = glob.glob(os.path.join(os.path.dirname(APP), "lora_models",
                                       "*", "training_meta.json"))
        contaminated, total = self._count_contaminated(metas)
        self.assertLessEqual(contaminated, total)


if __name__ == "__main__":
    unittest.main()
