from importlib import import_module
from pathlib import Path
import sys
import unittest


def load_contracts():
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    return import_module("prototail_gcd.contracts")


contracts = load_contracts()


class ContractTests(unittest.TestCase):
    def test_single_branch_baseline_output(self):
        output = contracts.ModelOutput(
            branches={
                "student": contracts.BranchOutput(
                    global_features=object(),
                    logits=object(),
                )
            }
        )

        self.assertIn("student", output.branches)
        self.assertIsNone(output.branches["student"].patch_tokens)

    def test_two_branch_baseline_can_expose_local_features(self):
        output = contracts.ModelOutput(
            branches={
                "classification": contracts.BranchOutput(
                    global_features=object(),
                    logits=object(),
                ),
                "contrastive": contracts.BranchOutput(
                    global_features=object(),
                    patch_tokens=object(),
                    projection=object(),
                ),
            }
        )

        self.assertIn("classification", output.branches)
        self.assertIsNotNone(output.branches["contrastive"].patch_tokens)

    def test_unknown_training_target_is_explicit(self):
        self.assertEqual(contracts.UNKNOWN_TARGET, -1)

    def test_pseudo_label_keeps_sample_identity_and_source(self):
        candidate = contracts.PseudoLabelCandidate(
            sample_id="unlabeled-17",
            class_id=4,
            confidence=0.91,
            source="novel_consensus",
        )

        self.assertEqual(candidate.sample_id, "unlabeled-17")
        self.assertEqual(candidate.source, "novel_consensus")


if __name__ == "__main__":
    unittest.main()
