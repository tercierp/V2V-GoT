#!/usr/bin/env python3
"""Generate lane-count descriptions from sat_images via a vision model."""
import argparse
import base64
import json
import mimetypes
import os
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from openai import OpenAI
    from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai. Install with: pip3 install openai"
    ) from exc

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SPLITS = [
    "train_01",
    "train_02",
    "train_03",
    "train_04",
    "train_05",
    "train_06",
    "train_07",
    "train_08",
    "test_01",
    "test_02",
    "test_03",
]

SYSTEM_PROMPT = (
    "You are a vision analyst. The vehicle is centered in the image and is heading "
    "toward the top of the image. Count the number of lanes visible around the vehicle "
    "and describe the scene briefly. Omit cars and vehicles from the description. "
    "Return JSON only."
)

USER_PROMPT = (
    "Return a JSON object with keys: "
    "lanes (integer), description (short phrase), confidence (0-1)."
)


def load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def iter_images(input_root: str, splits: List[str]) -> Iterable[Tuple[str, str, str, str]]:
    for split in splits:
        split_dir = os.path.join(input_root, split)
        if not os.path.isdir(split_dir):
            continue
        for sequence in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, sequence)
            if not os.path.isdir(seq_dir):
                continue
            for frame in sorted(os.listdir(seq_dir)):
                frame_dir = os.path.join(seq_dir, frame)
                if not os.path.isdir(frame_dir):
                    continue
                for filename in sorted(os.listdir(frame_dir)):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_EXTS:
                        continue
                    full_path = os.path.join(frame_dir, filename)
                    yield split, sequence, frame, full_path


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    encoded = encode_image_base64(path)
    return f"data:{mime};base64,{encoded}"


def extract_json(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_messages(data_url: str, image_mode: str) -> List[Dict]:
    if image_mode == "prompt":
        prompt = (
            f"{SYSTEM_PROMPT} {USER_PROMPT}\n"
            "Here is the base64-encoded image data (data URL format):\n"
            f"{data_url}"
        )
        return [{"role": "user", "content": prompt}]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def call_with_retry(
    client: OpenAI,
    model: str,
    data_url: str,
    max_retries: int,
    image_mode: str,
    reasoning_effort: Optional[str] = None,
    enable_thinking: bool = False,
) -> Dict:
    delay = 2
    messages = build_messages(data_url, image_mode)

    kwargs: Dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if enable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            return {"raw": text, "parsed": extract_json(text)}
        except BadRequestError as exc:
            message = str(exc)
            if "image_url" in message and "unknown variant" in message:
                raise RuntimeError(
                    "This model or endpoint does not accept image inputs. "
                    "Try --image-mode prompt or use a vision-capable model."
                ) from exc
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("Unexpected retry loop exit")


def load_processed(output_path: str) -> Set[str]:
    if not os.path.exists(output_path):
        return set()
    processed: Set[str] = set()
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel_path = record.get("rel_path")
            if rel_path:
                processed.add(rel_path)
    return processed


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(
        description="Generate lane-count descriptions from sat_images via a vision model."
    )
    parser.add_argument(
        "--input-root",
        default=os.path.join(script_dir, "sat_images"),
        help="Root folder containing sat_images splits.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "sat_images_descriptions.jsonl"),
        help="Path to JSONL output file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Vision-capable model.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split folders to process. Default: all train/test splits.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum number of images to process. Use 0 for no limit.",
    )
    parser.add_argument(
        "--env-file",
        default=os.path.join(repo_root, ".env"),
        help="Path to .env file containing OPENAI_API_KEY or DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL.",
    )
    parser.add_argument(
        "--image-mode",
        default="image_url",
        choices=["prompt", "image_url"],
        help="How to send image data: prompt (base64 text) or image_url.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the output JSONL.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient API errors.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["low", "medium", "high", "none"],
        help="Reasoning effort level for DeepSeek models. Use 'none' to omit.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable the thinking extra_body flag (DeepSeek only).",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_key = openai_key or deepseek_key
    if not api_key:
        raise SystemExit("OPENAI_API_KEY or DEEPSEEK_API_KEY is not set. Add it to .env or environment.")

    splits = args.splits if args.splits else DEFAULT_SPLITS
    input_root = os.path.abspath(args.input_root)
    output_path = os.path.abspath(args.output)

    processed = load_processed(output_path) if args.resume else set()

    if openai_key and not deepseek_key:
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    else:
        base_url = args.base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    client = OpenAI(api_key=api_key, base_url=base_url)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    use_deepseek_params = bool(deepseek_key) and not openai_key
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    if not use_deepseek_params:
        reasoning_effort = None
    enable_thinking = (not args.no_thinking) if use_deepseek_params else False

    written = 0
    with open(output_path, "a", encoding="utf-8") as out_handle:
        for split, sequence, frame, path in iter_images(input_root, splits):
            rel_path = os.path.relpath(path, input_root)
            if rel_path in processed:
                continue
            if args.max_images and written >= args.max_images:
                break

            data_url = image_to_data_url(path)
            result = call_with_retry(
                client,
                args.model,
                data_url,
                args.max_retries,
                image_mode=args.image_mode,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
            parsed = result.get("parsed") or {}

            record = {
                "split": split,
                "sequence": sequence,
                "frame": frame,
                "filename": os.path.basename(path),
                "rel_path": rel_path,
                "lanes": parsed.get("lanes"),
                "description": parsed.get("description"),
                "confidence": parsed.get("confidence"),
                "raw": result.get("raw"),
            }
            out_handle.write(json.dumps(record) + "\n")
            out_handle.flush()
            written += 1

    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
