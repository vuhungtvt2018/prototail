import importlib.util
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
import ssl
import sys
from unittest import mock
from urllib.error import URLError


HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeResponse:
    """Minimal urlopen stand-in: context manager with chunked ``read``."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            data, self._data = self._data, b""
        else:
            data, self._data = self._data[:size], self._data[size:]
        return data

    def close(self):
        self._data = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@unittest.skipUnless(HAS_TORCH, "Install torch to test the backbone loader")
class DinoBackboneCertificateFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.module = import_module("prototail_gcd.models.dino_backbone")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.hub_dir = Path(self._tmp.name)
        self.expected = self.hub_dir / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        self.weights_bytes = b"fake-dinov2-weights"

    def test_verified_download_writes_cache_file(self):
        response = _FakeResponse(self.weights_bytes)
        with mock.patch.object(
            self.module.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            staged = self.module.prefetch_checkpoint(
                "dinov2_vitb14", pretrained=True, hub_dir=self.hub_dir
            )
        self.assertEqual(staged, self.expected)
        self.assertEqual(self.expected.read_bytes(), self.weights_bytes)
        args, kwargs = urlopen.call_args
        self.assertEqual(args[0], self.module._WEIGHT_URLS["dinov2_vitb14"])
        self.assertNotIn("context", kwargs)
        self.assertFalse(self.expected.with_name(self.expected.name + ".part").exists())

    def test_ssl_certificate_error_retries_unverified(self):
        expired = ssl.SSLCertVerificationError(
            1, "certificate verify failed: certificate has expired"
        )
        response = _FakeResponse(self.weights_bytes)
        contexts = []

        def urlopen_side_effect(url, timeout=None, context=None):
            contexts.append(context)
            if context is None:
                raise URLError(expired)
            return response

        with mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=urlopen_side_effect
        ):
            staged = self.module.prefetch_checkpoint(
                "dinov2_vitb14", pretrained=True, hub_dir=self.hub_dir
            )
        self.assertEqual(staged, self.expected)
        self.assertEqual(self.expected.read_bytes(), self.weights_bytes)
        self.assertEqual(len(contexts), 2)
        self.assertIsNone(contexts[0])
        self.assertEqual(contexts[1].verify_mode, ssl.CERT_NONE)

    def test_persistent_failure_returns_none_and_cleans_part(self):
        with mock.patch.object(
            self.module.urllib.request,
            "urlopen",
            side_effect=URLError(OSError("no route to host")),
        ):
            staged = self.module.prefetch_checkpoint(
                "dinov2_vitb14", pretrained=True, hub_dir=self.hub_dir
            )
        self.assertIsNone(staged)
        self.assertFalse(self.expected.exists())
        self.assertFalse(
            self.expected.with_name(self.expected.name + ".part").exists()
        )

    def test_not_pretrained_returns_none(self):
        self.assertIsNone(
            self.module.prefetch_checkpoint(
                "dinov2_vitb14", pretrained=False, hub_dir=self.hub_dir
            )
        )

    def test_load_backbone_prefetches_before_hub_load(self):
        with mock.patch.object(
            self.module, "prefetch_checkpoint"
        ) as prefetch, mock.patch.object(
            self.module.torch.hub, "load", return_value="model"
        ) as hub_load:
            result = self.module.load_backbone("dinov2_vitb14", pretrained=True)
        self.assertEqual(result, "model")
        prefetch.assert_called_once_with("dinov2_vitb14", pretrained=True)
        hub_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
