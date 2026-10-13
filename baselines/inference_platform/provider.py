"""Provider adapters for Role-STVG inference.

Only provider-required fields are sent.  In particular, this module must not
grow model-tuning parameters: the benchmark protocol is provider defaults.
"""

from __future__ import annotations

import base64
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Protocol


class ProviderAdapter(Protocol):
    def infer(self, model: str, video_path: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        """Return raw text and provider-reported usage."""


def video_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{payload}"


class OpenAICompatibleAdapter:
    """OpenAI-compatible chat-completions adapter used by the ZhiPu gateway."""

    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: float = 600.0,
                 video_url_factory: Callable[[Path], str] | None = None):
        from openai import OpenAI

        # Timeout is transport behavior, not a model-facing inference parameter.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.video_url_factory = video_url_factory

    def infer(self, model: str, video_path: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        # Deliberately omit temperature, top_p, seed, max_tokens, response_format,
        # stream, visual sampling controls, and every other optional model knob.
        video_url = (
            self.video_url_factory(video_path)
            if self.video_url_factory is not None else video_data_url(video_path)
        )
        response = self.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            raw = content
        elif content is None:
            raw = ""
        else:
            raw = str(content)
        usage = response.usage
        if usage is None:
            usage_dict: dict[str, Any] = {}
        elif hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump(exclude_none=True)
        elif isinstance(usage, dict):
            usage_dict = dict(usage)
        else:
            usage_dict = {}
        return raw, usage_dict


class OssSignedHttpsVideoUrl:
    """Create a short-lived HTTPS URL for the corresponding frozen OSS object."""

    def __init__(self, *, local_root: Path, oss_root: str, expiry_duration: str = "2h",
                 ossutil: Path | None = None, config_file: Path | None = None):
        self.local_root = local_root.expanduser().resolve()
        self.oss_root = oss_root.rstrip("/")
        if not self.oss_root.startswith("oss://"):
            raise ValueError("signed-video OSS root must start with oss://")
        self.expiry_duration = expiry_duration
        self.ossutil = (ossutil or Path.home() / ".local/bin/ossutil").expanduser()
        self.config_file = (config_file or Path.home() / ".ossutilconfig").expanduser()

    def __call__(self, video_path: Path) -> str:
        relative = video_path.resolve().relative_to(self.local_root).as_posix()
        uri = f"{self.oss_root}/{relative}"
        # The installed ossutil 2.x expects the encrypted legacy config under
        # its newer section/key spelling. The temporary copy is mode 0600 and
        # is removed before this function returns.
        config = self.config_file.read_text(encoding="utf-8")
        config = re.sub(r"(?m)^\[default\]$", "[profile region_only]", config)
        config = re.sub(r"(?m)^\[profile default\]$", "[default]", config)
        config = re.sub(r"(?m)^accessKeyId=", "accessKeyID=", config)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as temporary:
            temporary.write(config)
            temporary.flush()
            completed = subprocess.run(
                [str(self.ossutil), "presign", uri, "--expires-duration",
                 self.expiry_duration, "--config-file", temporary.name],
                check=True, capture_output=True, text=True,
            )
        match = re.search(r"https://[^\s]+", completed.stdout)
        if not match:
            raise RuntimeError("ossutil did not return an HTTPS URL")
        return match.group(0)
