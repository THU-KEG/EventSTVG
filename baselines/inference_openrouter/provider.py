from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2


class OpenRouterEmptyResponseError(ConnectionError):
    """Retryable provider response that contains no usable completion choice."""


def _response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise OpenRouterEmptyResponseError(
            "OpenRouter returned no completion choices; retrying as a transient provider failure"
        )
    message = getattr(choices[0], "message", None)
    if message is None:
        raise OpenRouterEmptyResponseError(
            "OpenRouter returned a completion choice without a message; "
            "retrying as a transient provider failure"
        )
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ("" if content is None else str(content))


class OpenRouterFrameAdapter:
    """OpenRouter chat-completions adapter sending sampled JPEG frames.

    OpenRouter vision models accept ``image_url`` parts, while they generally do
    not accept ``video_url``. Frames are sampled at a fixed interval and labelled
    with their timestamp so temporal answers remain relative to the source video.
    """

    def __init__(self, *, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 timeout_seconds: float = 600.0, fps: float = 2.0,
                 max_frames: int = 1200, image_detail: str | None = "low"):
        # Import lazily so CLI help/manifest validation works before optional
        # runtime dependencies are installed.
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds,
                             default_headers={"HTTP-Referer": "https://openrouter.ai",
                                              "X-Title": "Role-STVG benchmark"})
        self.fps = fps
        self.max_frames = max_frames
        self.image_detail = image_detail

    def _frames(self, path: Path) -> list[tuple[float, str]]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if frame_count else 0.0
        result: list[tuple[float, str]] = []
        target_step = 1.0 / self.fps
        target_index = 0
        target_timestamp = 0.0
        index = 0
        while (len(result) < self.max_frames and target_index < frame_count
               and target_timestamp <= max(0.0, (frame_count - 1) / source_fps)):
            ok, frame = capture.read()
            if not ok:
                break
            wanted = round(target_timestamp * source_fps)
            if index >= min(wanted, max(0, frame_count - 1)):
                frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
                ok_jpg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    result.append((target_timestamp, base64.b64encode(encoded.tobytes()).decode("ascii")))
                target_index += 1
                target_timestamp = target_index * target_step
            index += 1
        capture.release()
        return result

    def infer(self, model: str, video_path: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        frames = self._frames(video_path)
        content: list[dict[str, Any]] = [{"type": "text", "text":
            "Analyze the following ordered video frames. Timestamps are seconds from video start.\n"}]
        for timestamp, payload in frames:
            content.append({"type": "text", "text": f"Frame at {timestamp:.2f}s:"})
            image_url = {"url": f"data:image/jpeg;base64,{payload}"}
            if self.image_detail is not None:
                image_url["detail"] = self.image_detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            extra_body={"provider": {"order": ["OpenAI"], "allow_fallbacks": False}},
        )
        raw = _response_content(response)
        usage = response.usage
        if usage is None: usage_dict = {}
        elif hasattr(usage, "model_dump"): usage_dict = usage.model_dump(exclude_none=True)
        elif isinstance(usage, dict): usage_dict = dict(usage)
        else: usage_dict = {}
        metadata = getattr(response, "model_extra", None)
        if isinstance(metadata, dict):
            if metadata.get("provider") is not None:
                usage_dict["serving_provider"] = metadata["provider"]
            if metadata.get("generation_id") is not None:
                usage_dict["generation_id"] = metadata["generation_id"]
        usage_dict["video_input"] = {"mode": "image_url_sequence", "frame_sample_fps": self.fps,
                                      "frame_sample_max_frames": self.max_frames,
                                      "frame_resize_width": 512, "frame_resize_height": 288,
                                      "frame_jpeg_quality": 80,
                                      "image_detail": self.image_detail,
                                      "sampled_frame_count": len(frames)}
        return raw, usage_dict


class OpenRouterNativeVideoAdapter:
    """OpenRouter adapter sending the original MP4 as one ``video_url`` part."""

    def __init__(self, *, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 timeout_seconds: float = 1200.0, provider_order: list[str] | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds,
                             default_headers={"HTTP-Referer": "https://openrouter.ai",
                                              "X-Title": "Role-STVG benchmark"})
        self.provider_order = provider_order

    def infer(self, model: str, video_path: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = base64.b64encode(video_path.read_bytes()).decode("ascii")
        video_url = f"data:video/mp4;base64,{payload}"
        request = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ]}],
        }
        if self.provider_order:
            request["extra_body"] = {"provider": {"order": self.provider_order,
                                                     "allow_fallbacks": False}}
        response = self.client.chat.completions.create(**request)
        raw = _response_content(response)
        usage = response.usage
        if usage is None: usage_dict: dict[str, Any] = {}
        elif hasattr(usage, "model_dump"): usage_dict = usage.model_dump(exclude_none=True)
        elif isinstance(usage, dict): usage_dict = dict(usage)
        else: usage_dict = {}
        metadata = getattr(response, "model_extra", None)
        if isinstance(metadata, dict):
            for key in ("provider", "provider_name", "generation_id", "cost", "cost_details"):
                if metadata.get(key) is not None: usage_dict[key] = metadata[key]
        usage_dict["video_input"] = {"mode": "video_url", "representation": "data:video/mp4;base64,<original MP4 bytes>",
                                      "source_video": str(video_path)}
        # No reasoning/thinking override is sent; this is solely an audit marker
        # indicating that provider/model defaults were requested.
        usage_dict["reasoning_requested"] = {"mode": "provider_default"}
        return raw, usage_dict
