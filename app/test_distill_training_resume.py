import os
import sys
import tempfile
import unittest

import torch
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.distill_train import get_resume_checkpoint


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index]),
            "labels": torch.tensor([index % 3]),
        }


class _Model(nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.dropout = nn.Dropout(0.35)
        self.projection = nn.Linear(8, 3)
        self.seen = seen

    def forward(self, input_ids=None, labels=None):
        self.seen.extend(input_ids.reshape(-1).detach().cpu().tolist())
        hidden = self.dropout(self.embedding(input_ids))
        logits = self.projection(hidden).mean(1)
        loss = nn.functional.cross_entropy(logits, labels.reshape(-1))
        return CausalLMOutput(loss=loss, logits=logits)


class _StopAtCheckpoint(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 3:
            control.should_save = True
            control.should_training_stop = True
        return control


def _training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_strategy="steps",
        save_steps=3,
        save_total_limit=2,
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        logging_strategy="no",
        seed=42,
        data_seed=42,
    )


def _new_model(seen):
    torch.manual_seed(99)
    return _Model(seen)


def _assert_nested_equal(testcase, expected, actual):
    testcase.assertEqual(type(expected), type(actual))
    if isinstance(expected, dict):
        testcase.assertEqual(set(expected), set(actual))
        for key in expected:
            _assert_nested_equal(testcase, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        testcase.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            _assert_nested_equal(testcase, left, right)
    elif torch.is_tensor(expected):
        testcase.assertTrue(torch.equal(expected, actual))
    else:
        testcase.assertEqual(expected, actual)


class DistillTrainingResumeTest(unittest.TestCase):
    def test_resume_preserves_training_state_rng_and_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_seen = []
            full_model = _new_model(full_seen)
            full_trainer = Trainer(
                model=full_model,
                args=_training_args(os.path.join(tmp, "full")),
                train_dataset=_Dataset(),
            )
            full_trainer.train()

            interrupted_seen = []
            interrupted_trainer = Trainer(
                model=_new_model(interrupted_seen),
                args=_training_args(os.path.join(tmp, "resumed")),
                train_dataset=_Dataset(),
                callbacks=[_StopAtCheckpoint()],
            )
            interrupted_trainer.train()
            checkpoint = os.path.join(tmp, "resumed", "checkpoint-3")
            self.assertEqual(
                {"model.safetensors", "optimizer.pt", "rng_state.pth",
                 "scheduler.pt", "trainer_state.json", "training_args.bin"},
                set(os.listdir(checkpoint)),
            )

            resumed_seen = []
            resumed_model = _new_model(resumed_seen)
            resumed_trainer = Trainer(
                model=resumed_model,
                args=_training_args(os.path.join(tmp, "resumed")),
                train_dataset=_Dataset(),
            )
            resumed_trainer.train(
                resume_from_checkpoint=get_resume_checkpoint(True))

            self.assertEqual(full_seen, interrupted_seen + resumed_seen)
            _assert_nested_equal(
                self, full_model.state_dict(), resumed_model.state_dict())
            _assert_nested_equal(
                self, full_trainer.optimizer.state_dict(),
                resumed_trainer.optimizer.state_dict())
            _assert_nested_equal(
                self, full_trainer.lr_scheduler.state_dict(),
                resumed_trainer.lr_scheduler.state_dict())

    def test_disabled_resume_starts_without_a_checkpoint(self):
        self.assertIsNone(get_resume_checkpoint(False))


if __name__ == "__main__":
    unittest.main()
