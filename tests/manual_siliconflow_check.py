"""Manual script to verify SiliconFlow API key works."""

from __future__ import annotations

import os

import requests
from config.settings import load_config

load_config()  # 会读取 .env 并写入 os.environ

BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("SILICONFLOW_API_KEY")
MODEL = os.getenv("SILICONFLOW_MODEL", "gpt4-mini")

if not API_KEY:
    print("[error] SILICONFLOW_API_KEY not set; check .env or environment variables.")
    raise SystemExit(1)

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

try:
    resp = requests.get(f"{BASE_URL}/models", headers=headers, timeout=30)
    print("Status:", resp.status_code)
    if resp.ok:
        data = resp.json()
        print("Models count:", len(data.get("data", [])))
        for item in data.get("data", [])[:5]:
            print("-", item.get("id"))
    else:
        print(resp.text[:500])

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "简要描述一只坐在窗台上的橘猫。"},
        ],
        "max_tokens": 120,
    }
    chat = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    print("Chat status:", chat.status_code)
    if chat.ok:
        data = chat.json()
        choice = data.get("choices", [{}])[0]
        print("Generation:", choice.get("message", {}).get("content"))
    else:
        print(chat.text[:500])
except Exception as exc:  # noqa: BLE001
    print("[error]", exc)
    raise
