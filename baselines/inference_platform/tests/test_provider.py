from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from inference_platform.provider import OpenAICompatibleAdapter
from inference_platform.core import transient_error
from inference_openrouter.provider import OpenRouterEmptyResponseError, _response_content


class ProviderTests(unittest.TestCase):
    def test_openrouter_empty_choices_are_explicitly_retryable(self):
        for choices in (None, []):
            with self.subTest(choices=choices):
                with self.assertRaises(OpenRouterEmptyResponseError) as raised:
                    _response_content(types.SimpleNamespace(choices=choices))
                self.assertTrue(transient_error(raised.exception))

    def test_openrouter_response_content_is_extracted(self):
        message = types.SimpleNamespace(content="{}")
        choice = types.SimpleNamespace(message=message)
        self.assertEqual(_response_content(types.SimpleNamespace(choices=[choice])), "{}")

    def test_request_contains_no_optional_model_parameters(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(content="{}")
                choice = types.SimpleNamespace(message=message)
                usage = types.SimpleNamespace(model_dump=lambda **kwargs: {"total_tokens": 3})
                return types.SimpleNamespace(choices=[choice], usage=usage)

        class Client:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=Completions())

        fake_openai = types.SimpleNamespace(OpenAI=Client)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            adapter = OpenAICompatibleAdapter(
                api_key="secret", base_url="https://example.invalid/v1"
            )
            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / "video.mp4"
                video.write_bytes(b"video")
                raw, usage = adapter.infer("model", video, "frozen prompt")

        self.assertEqual(raw, "{}")
        self.assertEqual(usage, {"total_tokens": 3})
        self.assertEqual(set(captured), {"model", "messages"})
        self.assertNotIn("stream", captured)
        self.assertEqual(captured["messages"][0]["content"][1]["text"], "frozen prompt")


if __name__ == "__main__":
    unittest.main()
