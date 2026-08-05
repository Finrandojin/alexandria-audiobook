import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from experiments import blinded_listening as blind
from experiments.provenance import file_sha256, input_sha256


def write_wav(path, frames=320):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


def provenance(script):
    return {"script": script, "git": {"harness_sha256": "a" * 64}}


class BlindedListeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=blind.REPO)
        self.root = Path(self.temp.name)
        self.audio = self.root / "source"
        self.audio.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name):
        path = self.audio / f"{name}.wav"
        write_wav(path)
        return os.path.relpath(path, blind.REPO)

    def documents(self):
        comparisons = []
        for line in range(4):
            comparisons.append({
                "arm_files": {arm: self.source(f"line{line}_{arm}")
                              for arm in ("none", "per_char", "per_line")}})
        instruction = {
            "status": "complete", "all_arms_rendered": True,
            "provenance": provenance("instruct_listening.py"),
            "comparisons": comparisons}
        casting = {
            "status": "complete", "published": True, "lines": 2,
            "provenance": provenance("casting_ab_audio.py"),
            "arms": {arm: {"path": self.source(f"cast_{arm}"),
                            "lines": [0, 1]}
                     for arm in ("current", "scene_aware")}}
        rows = []
        for row in range(3):
            controls = {}
            for arm in ("very_slow", "very_fast"):
                relative = self.source(f"control{row}_{arm}")
                controls[arm] = {
                    "file": relative,
                    "sha256": file_sha256(os.path.join(blind.REPO, relative))}
            rows.append({"duration_order_control_passes": True,
                         "instruction_controls": controls})
        controls = {"provenance": provenance("seed_instruction_controls.py"),
                    "rows": rows}
        return instruction, casting, controls

    def write_documents(self):
        paths = []
        for name, document in zip(
                ("instruction.json", "casting.json", "controls.json"),
                self.documents()):
            path = self.root / name
            path.write_text(json.dumps(document), encoding="utf-8")
            paths.append(str(path))
        return paths

    def build(self, suffix=""):
        instruction, casting, controls = self.write_documents()
        package = str(self.root / f"package{suffix}")
        public_path = str(self.root / f"public{suffix}.json")
        key_path = str(self.root / f"key{suffix}.json")
        public, key = blind.build_package(
            instruction, casting, controls, package, public_path, key_path, 42)
        return public, key, package, public_path, key_path

    def test_builds_eight_unlabeled_sets_and_strictly_revalidates(self):
        public, key, package, public_path, key_path = self.build()
        checked_public, checked_key = blind.validate_package(
            public_path, key_path, package)

        self.assertEqual(8, len(public["sets"]))
        self.assertEqual(public, checked_public)
        self.assertEqual(key, checked_key)
        serialized = json.dumps(public)
        for label in ("very_slow", "very_fast", "per_char", "per_line",
                      '"current"', '"scene_aware"'):
            self.assertNotIn(label, serialized)
        self.assertEqual(
            {sample["file"] for item in public["sets"] for sample in item["samples"]},
            set(os.listdir(package)))

    def test_same_seed_produces_the_same_concealed_mapping(self):
        _, first_key, _, _, _ = self.build("_one")
        _, second_key, _, _, _ = self.build("_two")
        self.assertEqual(first_key, second_key)

    def test_validation_rejects_changed_audio_and_extra_files(self):
        _, _, package, public_path, key_path = self.build()
        public = json.loads(Path(public_path).read_text(encoding="utf-8"))
        first = Path(package, public["sets"][0]["samples"][0]["file"])
        first.write_bytes(first.read_bytes()[:-1])
        with self.assertRaises(blind.ListeningPackageError):
            blind.validate_package(public_path, key_path, package)

        write_wav(first)
        public["sets"][0]["samples"][0]["sha256"] = file_sha256(first)
        mapped = json.loads(Path(key_path).read_text(encoding="utf-8"))
        mapped["sets"][0]["mapping"][first.name]["source_sha256"] = file_sha256(first)
        Path(key_path).write_text(json.dumps(mapped), encoding="utf-8")
        public["concealed_key_sha256"] = file_sha256(key_path)
        Path(public_path).write_text(json.dumps(public), encoding="utf-8")
        write_wav(Path(package, "unexpected.wav"))
        with self.assertRaisesRegex(blind.ListeningPackageError,
                                    "inventory changed"):
            blind.validate_package(public_path, key_path, package)

    def test_validation_rejects_changed_source_manifest(self):
        _, _, package, public_path, key_path = self.build()
        source = self.root / "instruction.json"
        source.write_text(source.read_text(encoding="utf-8") + "\n",
                          encoding="utf-8")
        with self.assertRaisesRegex(blind.ListeningPackageError,
                                    "source artifact changed"):
            blind.validate_package(public_path, key_path, package)

    def test_rejects_asymmetric_casting_arms(self):
        instruction, casting, controls = self.documents()
        casting["arms"]["scene_aware"]["lines"] = [0]
        with self.assertRaisesRegex(blind.ListeningPackageError,
                                    "identical lines"):
            blind._source_groups(instruction, casting, controls)

    def test_rejects_truncated_wav_during_full_decode(self):
        path = self.root / "truncated.wav"
        write_wav(path)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(blind.ListeningPackageError):
            blind._validate_wav(str(path))

    def test_rejects_source_without_provenance(self):
        path = self.root / "source.json"
        path.write_text('{"status":"complete"}', encoding="utf-8")
        with self.assertRaisesRegex(blind.ListeningPackageError, "no provenance"):
            blind._load_document(str(path))

    def test_failed_manifest_write_removes_all_new_outputs(self):
        instruction, casting, controls = self.write_documents()
        package = str(self.root / "package")
        public_path = str(self.root / "public.json")
        key_path = str(self.root / "key.json")
        from utils import atomic_json_write as real_write
        calls = 0

        def fail_second(document, path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected public-manifest failure")
            return real_write(document, path)

        with patch("utils.atomic_json_write", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                blind.build_package(instruction, casting, controls, package,
                                    public_path, key_path, 42)
        self.assertFalse(os.path.exists(package))
        self.assertFalse(os.path.exists(public_path))
        self.assertFalse(os.path.exists(key_path))

    def test_input_hashes_fail_loudly_for_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            input_sha256((str(self.root / "missing"),))


if __name__ == "__main__":
    unittest.main()
