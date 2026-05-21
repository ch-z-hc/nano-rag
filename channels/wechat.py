"""WeChat personal bot via ilinkai HTTP API.

Protocol reverse-engineered from ``@tencent-weixin/openclaw-weixin`` v1.0.3,
same as nanobot's implementation.
"""

import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path

import aiohttp
import qrcode

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
SESSION_FILE = Path(__file__).resolve().parent.parent / ".wechat_session.json"

# Protocol constants
ILINK_APP_ID = "bot"
WEIXIN_CHANNEL_VERSION = "2.1.1"
BASE_INFO = {"channel_version": WEIXIN_CHANNEL_VERSION}
ITEM_TEXT = 1
MESSAGE_TYPE_BOT = 2
MESSAGE_STATE_FINISH = 2
ERRCODE_SESSION_EXPIRED = -14


def _build_client_version(version: str) -> int:
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return ((major & 0xFF) << 16) | ((minor & 0xFF) << 8) | (patch & 0xFF)


ILINK_APP_CLIENT_VERSION = _build_client_version(WEIXIN_CHANNEL_VERSION)


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_session() -> dict:
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_session(data: dict) -> None:
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _random_wechat_uin() -> str:
    """Generate X-WECHAT-UIN header per request (matches reference)."""
    uint32 = int.from_bytes(os.urandom(4), "big")
    return base64.b64encode(str(uint32).encode()).decode()


def _make_headers(token: str = "", route_tag: str = "") -> dict:
    """Build per-request headers with fresh UIN each call (matches api.ts)."""
    headers = {
        "X-WECHAT-UIN": _random_wechat_uin(),
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if route_tag:
        headers["SKRouteTag"] = route_tag
    return headers


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


async def _api_get(
    sess: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    params: dict | None = None,
    token: str = "",
    route_tag: str = "",
) -> dict:
    url = f"{base_url}/{endpoint}"
    headers = _make_headers(token=token, route_tag=route_tag)
    async with sess.get(url, params=params, headers=headers) as resp:
        resp.raise_for_status()
        # API returns JSON but with application/octet-stream content-type
        text = await resp.text()
        return json.loads(text)


async def _api_post(
    sess: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    body: dict | None = None,
    token: str = "",
    route_tag: str = "",
) -> dict:
    url = f"{base_url}/{endpoint}"
    payload = body or {}
    if "base_info" not in payload:
        payload["base_info"] = BASE_INFO
    headers = _make_headers(token=token, route_tag=route_tag)
    async with sess.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        # API returns JSON but with application/octet-stream content-type
        text = await resp.text()
        return json.loads(text)


# ---------------------------------------------------------------------------
# QR Login
# ---------------------------------------------------------------------------


async def login_wechat() -> str | None:
    """Interactive WeChat login. Returns token on success."""
    cfg = _load_config()["wechat"]
    base_url = cfg["api_base"]
    route_tag = str(cfg.get("route_tag", "") or "")

    async with aiohttp.ClientSession() as sess:
        # 1. Fetch QR code
        data = await _api_get(
            sess, base_url, "ilink/bot/get_bot_qrcode",
            params={"bot_type": "3"},
            route_tag=route_tag,
        )
        qrcode_id = data.get("qrcode", "")
        qrcode_img = data.get("qrcode_img_content", "")
        if not qrcode_id:
            print(f"Failed to get QR code: {data}")
            return None

        scan_url = qrcode_img if qrcode_img.startswith("http") else qrcode_id

        # Print QR in terminal
        qr = qrcode.QRCode(border=1)
        qr.add_data(scan_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\nLogin URL: {scan_url}")
        print("\n请用微信扫描上方二维码，等待确认...")

        # 2. Poll for confirmation
        current_base_url = base_url
        for _ in range(120):
            await asyncio.sleep(1.5)
            try:
                status_data = await _api_get(
                    sess, current_base_url, "ilink/bot/get_qrcode_status",
                    params={"qrcode": qrcode_id},
                    route_tag=route_tag,
                )
            except Exception:
                await asyncio.sleep(1)
                continue

            status = status_data.get("status", "")

            if status == "scaned_but_not_confirm":
                print("已扫码，请在手机上确认登录...")
            elif status == "scaned_but_redirect":
                redirect_host = str(status_data.get("redirect_host", "") or "").strip()
                if redirect_host:
                    if not redirect_host.startswith("http"):
                        redirect_host = f"https://{redirect_host}"
                    current_base_url = redirect_host
            elif status == "confirmed":
                token = status_data.get("bot_token", "")
                bot_id = status_data.get("ilink_bot_id", "")
                user_id = status_data.get("ilink_user_id", "")
                new_base_url = status_data.get("baseurl", "")
                if token:
                    if new_base_url:
                        base_url = new_base_url
                    session_data = {
                        "token": token,
                        "bot_id": bot_id,
                        "user_id": user_id,
                        "base_url": base_url,
                        "get_updates_buf": "",
                        "context_tokens": {},
                        "login_time": time.time(),
                    }
                    _save_session(session_data)
                    print(f"登录成功! bot_id={bot_id} user_id={user_id}")
                    return token
                else:
                    print("登录确认但未收到 token")
                    return None
            elif status == "expired":
                print("二维码已过期，请重试")
                return None

        print("登录超时")
        return None


# ---------------------------------------------------------------------------
# Message polling
# ---------------------------------------------------------------------------


async def poll_messages(
    sess: aiohttp.ClientSession,
    base_url: str,
    session: dict,
    on_message,
):
    """Long-poll for new messages. Calls ``on_message`` for each text message."""
    token = session["token"]
    route_tag = str(_load_config()["wechat"].get("route_tag", "") or "")
    get_updates_buf = session.get("get_updates_buf", "")
    seen_ids: dict[str, None] = {}

    while True:
        try:
            body = {"get_updates_buf": get_updates_buf}
            data = await _api_post(
                sess, base_url, "ilink/bot/getupdates",
                body=body, token=token, route_tag=route_tag,
            )

            ret = data.get("ret", 0)
            errcode = data.get("errcode", 0)
            if (ret is not None and ret != 0) or (errcode is not None and errcode != 0):
                if errcode == ERRCODE_SESSION_EXPIRED or ret == ERRCODE_SESSION_EXPIRED:
                    print("[wechat] session expired, need re-login")
                    return
                print(f"[wechat] getupdates error: ret={ret} errcode={errcode}")
                await asyncio.sleep(3)
                continue

            # Update cursor
            new_buf = data.get("get_updates_buf", "")
            if new_buf:
                get_updates_buf = new_buf
                session["get_updates_buf"] = new_buf
                _save_session(session)

            msgs = data.get("msgs", []) or []
            for msg in msgs:
                # Skip bot's own messages
                if msg.get("message_type") == MESSAGE_TYPE_BOT:
                    continue

                msg_id = str(msg.get("message_id", "") or msg.get("seq", ""))
                if not msg_id:
                    msg_id = f"{msg.get('from_user_id', '')}_{msg.get('create_time_ms', '')}"

                if msg_id in seen_ids:
                    continue
                seen_ids[msg_id] = None
                if len(seen_ids) > 1000:
                    seen_ids.pop(next(iter(seen_ids)))

                from_user = msg.get("from_user_id", "")
                if not from_user:
                    continue

                # Cache context_token for reply
                ctx_token = msg.get("context_token", "")
                if ctx_token:
                    ctx_tokens = session.get("context_tokens", {})
                    ctx_tokens[from_user] = ctx_token
                    session["context_tokens"] = ctx_tokens
                    _save_session(session)

                # Parse text from item_list
                item_list = msg.get("item_list") or []
                content_parts = []
                for item in item_list:
                    if item.get("type") == ITEM_TEXT:
                        text = (item.get("text_item") or {}).get("text", "")
                        if text:
                            content_parts.append(text)

                content = "".join(content_parts)
                if content:
                    await on_message(from_user, content, msg_id)

            await asyncio.sleep(1.5)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[wechat poll error] {e}")
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Send text
# ---------------------------------------------------------------------------


async def send_text(
    sess: aiohttp.ClientSession,
    base_url: str,
    session: dict,
    to_user: str,
    text: str,
):
    """Send a text message to a WeChat user."""
    token = session["token"]
    route_tag = str(_load_config()["wechat"].get("route_tag", "") or "")
    ctx_tokens = session.get("context_tokens", {})
    ctx_token = ctx_tokens.get(to_user, "")

    client_id = f"nano-rag-{uuid.uuid4().hex[:12]}"

    weixin_msg = {
        "from_user_id": "",
        "to_user_id": to_user,
        "client_id": client_id,
        "message_type": MESSAGE_TYPE_BOT,
        "message_state": MESSAGE_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }
    if ctx_token:
        weixin_msg["context_token"] = ctx_token

    body = {"msg": weixin_msg}

    try:
        data = await _api_post(
            sess, base_url, "ilink/bot/sendmessage",
            body=body, token=token, route_tag=route_tag,
        )
        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        if (ret is not None and ret != 0) or (errcode is not None and errcode != 0):
            print(f"[wechat send fail] ret={ret} errcode={errcode}: {data.get('errmsg', '')}")
    except Exception as e:
        print(f"[wechat send error] {e}")


# ---------------------------------------------------------------------------
# Run bot
# ---------------------------------------------------------------------------


async def run_wechat_bot(on_message):
    """Main entry: restore session, start polling.

    ``on_message(from_user, content, msg_id) -> str | None``
    Return a string to reply.
    """
    session = _load_session()
    token = session.get("token", "")
    base_url = session.get("base_url", _load_config()["wechat"]["api_base"])

    if not token:
        print("No saved session. Logging in...")
        token = await login_wechat()
        if not token:
            print("Login failed.")
            return
        session = _load_session()
        base_url = session.get("base_url", base_url)

    print(f"WeChat bot running (bot_id={session.get('bot_id', '?')})...")

    async with aiohttp.ClientSession() as sess:
        async def handler(from_user, content, msg_id):
            try:
                reply = await on_message(from_user, content, msg_id)
                if reply:
                    await send_text(sess, base_url, session, from_user, reply)
            except Exception as e:
                print(f"[handler error] {e}")

        await poll_messages(sess, base_url, session, handler)
