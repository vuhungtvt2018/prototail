from importlib import import_module
from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock


def load_module():
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    return import_module("prototail_gcd.models.dino_backbone")


dino_backbone = load_module()

ENTRYPOINT = "dinov2_vitb14"
HUBCONF_TEXT = "from dinov2.hub.backbones import dinov2_vitb14, dinov2_vits14\n"


class _TempDirsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmp.name)
        self.weights_bytes = b"fake-dinov2-weights"
        self.addCleanup(self._tmp.cleanup)

    def make_repo(self, parent: Path, name: str = "dinov2_repo", *, entrypoint: bool = True) -> Path:
        repo = parent / name
        repo.mkdir(parents=True, exist_ok=True)
        text = HUBCONF_TEXT if entrypoint else "from other.hub import something_else\n"
        (repo / "hubconf.py").write_text(text, encoding="utf-8")
        return repo


class FindLocalRepoTests(_TempDirsTestCase):
    def test_found_in_hub_dir_cache(self):
        repo = self.make_repo(self.tmp_root, name="facebookresearch_dinov2_main")
        found = dino_backbone.find_local_repo(ENTRYPOINT, hub_dir=self.tmp_root)
        self.assertEqual(found, repo)

    def test_rejects_hubconf_without_entrypoint(self):
        self.make_repo(
            self.tmp_root, name="facebookresearch_dinov2_main", entrypoint=False
        )
        self.assertIsNone(dino_backbone.find_local_repo(ENTRYPOINT, hub_dir=self.tmp_root))

    def test_found_by_scan_of_extra_root(self):
        repo = self.make_repo(self.tmp_root / "dataset-mount" / "dinov2_repo")
        found = dino_backbone.find_local_repo(
            ENTRYPOINT, hub_dir=self.tmp_root / "empty_hub", extra_roots=[self.tmp_root]
        )
        self.assertEqual(found, repo)

    def test_env_override_wins(self):
        repo = self.make_repo(self.tmp_root / "explicit")
        with mock.patch.dict(os.environ, {dino_backbone._REPO_ENV_VAR: str(repo)}):
            found = dino_backbone.find_local_repo(ENTRYPOINT, hub_dir=self.tmp_root)
        self.assertEqual(found, repo)


class EnsureOfflineCheckpointTests(_TempDirsTestCase):
    def setUp(self):
        super().setUp()
        self.hub_dir = self.tmp_root / "hub"

    def test_returns_none_without_pretrained(self):
        self.assertIsNone(
            dino_backbone.ensure_offline_checkpoint(
                ENTRYPOINT, pretrained=False, hub_dir=self.hub_dir
            )
        )

    def test_copies_from_extra_root_scan(self):
        bundle = self.tmp_root / "bundle"
        (bundle / "checkpoints").mkdir(parents=True)
        weights = bundle / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        weights.write_bytes(self.weights_bytes)

        staged = dino_backbone.ensure_offline_checkpoint(
            ENTRYPOINT,
            pretrained=True,
            hub_dir=self.hub_dir,
            extra_roots=[bundle],
        )
        expected = self.hub_dir / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        self.assertEqual(staged, expected)
        self.assertTrue(expected.is_file())
        self.assertEqual(expected.read_bytes(), self.weights_bytes)

    def test_fuzzy_name_and_reg_guard(self):
        bundle = self.tmp_root / "bundle"
        (bundle / "weights").mkdir(parents=True)
        reg_file = bundle / "weights" / "dinov2_vitb14_reg4_pretrain.pth"
        reg_file.write_bytes(b"reg-weights")

        self.assertIsNone(
            dino_backbone.ensure_offline_checkpoint(
                ENTRYPOINT, pretrained=True, hub_dir=self.hub_dir, extra_roots=[bundle]
            ),
            "reg4 checkpoint must not be staged for the non-reg backbone",
        )

        renamed = bundle / "weights" / "dinov2_vitb14_renamed.pth"
        renamed.write_bytes(self.weights_bytes)
        staged = dino_backbone.ensure_offline_checkpoint(
            ENTRYPOINT, pretrained=True, hub_dir=self.hub_dir, extra_roots=[bundle]
        )
        self.assertIsNotNone(staged)
        self.assertEqual(staged.read_bytes(), self.weights_bytes)

    def test_env_override(self):
        weights = self.tmp_root / "dinov2_vitb14_pretrain.pth"
        weights.write_bytes(self.weights_bytes)
        with mock.patch.dict(
            os.environ, {dino_backbone._WEIGHTS_ENV_VAR: str(weights)}
        ):
            staged = dino_backbone.ensure_offline_checkpoint(
                ENTRYPOINT, pretrained=True, hub_dir=self.hub_dir
            )
        self.assertEqual(staged, self.hub_dir / "checkpoints" / "dinov2_vitb14_pretrain.pth")


class LoadBackboneOfflineTests(_TempDirsTestCase):
    def test_fast_fail_offline_without_assets(self):
        with mock.patch.object(dino_backbone, "find_local_repo", return_value=None), \
            mock.patch.object(dino_backbone, "probe_internet", return_value=False), \
            mock.patch.object(dino_backbone.torch.hub, "load") as hub_load:
            with self.assertRaises(RuntimeError) as ctx:
                dino_backbone.load_backbone(ENTRYPOINT)
        hub_load.assert_not_called()
        self.assertIn("dinov2_offline", str(ctx.exception))

    def test_uses_local_repo_and_local_source(self):
        repo = self.make_repo(self.tmp_root / "repo")
        checkpoint = self.tmp_root / "dinov2_vitb14_pretrain.pth"
        checkpoint.write_bytes(self.weights_bytes)
        sentinel = object()
        with mock.patch.object(dino_backbone, "find_local_repo", return_value=repo), \
            mock.patch.object(
                dino_backbone,
                "ensure_offline_checkpoint",
                return_value=checkpoint,
            ), \
            mock.patch.object(
                dino_backbone.torch.hub, "load", return_value=sentinel
            ) as hub_load:
            result = dino_backbone.load_backbone(ENTRYPOINT, pretrained=True)
        self.assertIs(result, sentinel)
        args, kwargs = hub_load.call_args
        self.assertEqual(args[0], str(repo))
        self.assertEqual(args[1], ENTRYPOINT)
        self.assertEqual(kwargs.get("source"), "local")
        self.assertEqual(kwargs.get("pretrained"), True)

    def test_offline_repo_without_weights_raises_with_instructions(self):
        repo = self.make_repo(self.tmp_root / "repo")
        with mock.patch.object(dino_backbone, "find_local_repo", return_value=repo), \
            mock.patch.object(
                dino_backbone, "ensure_offline_checkpoint", return_value=None
            ), \
            mock.patch.object(dino_backbone, "probe_internet", return_value=False), \
            mock.patch.object(dino_backbone.torch.hub, "load") as hub_load:
            with self.assertRaises(RuntimeError) as ctx:
                dino_backbone.load_backbone(ENTRYPOINT, pretrained=True)
        hub_load.assert_not_called()
        self.assertIn("PROTOTAIL_DINO_WEIGHTS", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
