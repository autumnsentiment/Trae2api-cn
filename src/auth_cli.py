"""
auth_cli.py - OAuth device authorization flow (mimics Trae CLI)

The Trae desktop client uses a device authorization flow:
1. POST /cloudide/api/v3/trae/oauth/Authorize
   -> Returns { Result: { device_code, user_code, verification_uri, expires_in, interval } }
2. User visits the verification_uri (or scans QR code) to authorize
3. POST /cloudide/api/v3/trae/oauth/GetUserToken (polling)
   -> Returns { Result: { Token, RefreshToken, TokenExpireAt, ... } }
   -> Or { ResponseMetadata: { Error: { Code: "AuthorizationPending" } } }

This module implements the full flow so the relay can obtain its own credentials
without needing a pre-existing Trae installation or manual token paste.

WARNING: the /cloudide/api/v3/trae/oauth/Authorize endpoint currently returns 404,
so this flow is not usable. Keep the module only as a reference; prefer
TRAE_AUTH_SOURCE=cli or manual/env credentials instead.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CLIENT_ID = "ono9krqynydwx5"
POLL_INTERVAL = 3  # default seconds between polls
MAX_POLL_TIME = 300  # give up after 5 minutes


@dataclass
class DeviceAuthResponse:
    """Result from the Authorize endpoint."""
    device_code: str = ""
    user_code: str = ""
    verification_uri: str = ""
    verification_uri_complete: str = ""
    expires_in: int = 300
    interval: int = 3
    qr_code: str = ""  # base64 PNG of QR code, if returned


@dataclass
class TokenResult:
    """Result from the GetUserToken endpoint."""
    token: str = ""
    refresh_token: str = ""
    user_id: str = ""
    tenant_id: str = ""
    expires_at: int = 0  # unix ms timestamp
    refresh_expires_at: int = 0
    error_code: str = ""
    error_message: str = ""


async def start_device_authorization(
    host: str = "https://trae-api-cn.mchost.guru",
    client_id: str = DEFAULT_CLIENT_ID,
    edition: str = "cn",
) -> DeviceAuthResponse:
    """
    POST /cloudide/api/v3/trae/oauth/Authorize
    Initiates the device authorization flow.
    """
    url = f"{host.rstrip('/')}/cloudide/api/v3/trae/oauth/Authorize"
    payload = {
        "ClientID": client_id,
        "ClientSecret": "-",
        "AuthType": "device",
        "Edition": edition,
    }

    logger.info("auth-cli: POST %s", url)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        text = resp.text

    if resp.status_code != 200:
        raise RuntimeError(f"Authorize failed [{resp.status_code}]: {text[:500]}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Authorize: invalid JSON response: {text[:500]}")

    # Check for error envelope
    err = data.get("ResponseMetadata", {}).get("Error")
    if err:
        raise RuntimeError(f"Authorize error: {err.get('Code')}: {err.get('Message')}")

    result = data.get("Result") or data.get("result") or data
    da = DeviceAuthResponse(
        device_code=result.get("device_code") or result.get("DeviceCode") or "",
        user_code=result.get("user_code") or result.get("UserCode") or "",
        verification_uri=result.get("verification_uri") or result.get("VerificationUri") or "",
        verification_uri_complete=result.get("verification_uri_complete") or result.get("VerificationUriComplete") or "",
        expires_in=result.get("expires_in") or result.get("ExpiresIn") or 300,
        interval=result.get("interval") or result.get("Interval") or 3,
        qr_code=result.get("qrcode") or result.get("qr_code") or result.get("QrCode") or "",
    )

    if not da.device_code:
        raise RuntimeError(f"Authorize: no device_code in response: {text[:500]}")

    logger.info("auth-cli: device_code=%s..., user_code=%s, uri=%s",
                da.device_code[:12], da.user_code, da.verification_uri)
    return da


async def poll_user_token(
    host: str,
    client_id: str,
    device_code: str,
    on_progress: Optional[callable] = None,
) -> TokenResult:
    """
    POST /cloudide/api/v3/trae/oauth/GetUserToken
    Polls until the user authorizes the device or the code expires.

    on_progress callback receives (attempt, elapsed_seconds, response_data) for logging.
    """
    url = f"{host.rstrip('/')}/cloudide/api/v3/trae/oauth/GetUserToken"
    start = time.time()
    deadline = start + MAX_POLL_TIME

    attempt = 0
    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < deadline:
            attempt += 1
            payload = {
                "ClientID": client_id,
                "ClientSecret": "-",
                "DeviceCode": device_code,
            }

            try:
                resp = await client.post(url, json=payload)
                text = resp.text

                if resp.status_code != 200:
                    raise RuntimeError(f"GetUserToken HTTP [{resp.status_code}]: {text[:500]}")

                data = json.loads(text)

                # Check for pending status
                err = data.get("ResponseMetadata", {}).get("Error", {})
                err_code = (err.get("Code") or "").lower()
                if err_code:
                    if "pending" in err_code or "authorization" in err_code:
                        elapsed = time.time() - start
                        if on_progress:
                            on_progress(attempt, elapsed, data)
                        # Wait the recommended interval before retrying
                        await asyncio.sleep(data.get("Result", {}).get("interval", POLL_INTERVAL))
                        continue
                    # Other errors are terminal
                    tr = TokenResult(error_code=err.get("Code", ""), error_message=err.get("Message", ""))
                    logger.error("auth-cli: GetUserToken error: %s", tr.error_code)
                    return tr

                result = data.get("Result") or data.get("result") or data
                token = result.get("Token") or result.get("token") or ""
                if not token:
                    elapsed = time.time() - start
                    if on_progress:
                        on_progress(attempt, elapsed, data)
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                tr = TokenResult(
                    token=token,
                    refresh_token=result.get("RefreshToken") or result.get("refreshToken") or "",
                    user_id=result.get("UserID") or result.get("userId") or result.get("UserId") or "",
                    tenant_id=result.get("TenantID") or result.get("tenantId") or "",
                    expires_at=result.get("TokenExpireAt") or result.get("tokenExpireAt") or 0,
                    refresh_expires_at=result.get("RefreshExpireAt") or result.get("refreshExpireAt") or 0,
                )
                logger.info("auth-cli: token obtained (user=%s, expires_at=%s)", tr.user_id, tr.expires_at)
                return tr

            except httpx.RequestError as e:
                logger.warning("auth-cli: poll attempt %d error: %s", attempt, e)
                await asyncio.sleep(POLL_INTERVAL * 2)
                continue

    return TokenResult(error_code="expired", error_message="Device authorization timed out")


async def device_authorization_flow(
    host: str = "https://trae-api-cn.mchost.guru",
    client_id: str = DEFAULT_CLIENT_ID,
    edition: str = "cn",
    on_progress: Optional[callable] = None,
) -> TokenResult:
    """
    Full device authorization flow:
    1. Call Authorize to get device + user code
    2. Poll GetUserToken until user authorizes
    3. Return the TokenResult

    The caller should display the verification_uri and user_code to the user.
    """
    da = await start_device_authorization(host, client_id, edition)

    # Print instructions for the user
    logger.info("=" * 60)
    logger.info("Trae CN Device Authorization")
    logger.info("=" * 60)
    logger.info("1. Open your browser and visit:")
    logger.info("   %s", da.verification_uri_complete or da.verification_uri)
    logger.info("2. Enter the code: %s", da.user_code)
    logger.info("3. Authorize the application")
    logger.info("=" * 60)
    logger.info("")

    # If we have a QR code url, log it
    if da.qr_code:
        logger.info("QR code available (base64 PNG, %d bytes)", len(da.qr_code))

    print(f"\n{'='*60}")
    print(f"  Trae CN Device Authorization")
    print(f"{'='*60}")
    print(f"  1. Open your browser and visit:")
    print(f"     {da.verification_uri_complete or da.verification_uri}")
    print(f"  2. Enter the code: {da.user_code}")
    print(f"  3. Authorize the application")
    print(f"{'='*60}\n")

    # Poll for the token
    result = await poll_user_token(host, client_id, da.device_code, on_progress)
    return result


async def exchange_refresh_token(
    host: str,
    client_id: str,
    refresh_token: str,
    user_id: str = "",
) -> TokenResult:
    """
    Exchange a refresh token for a new access token.
    Same endpoint as the existing auth.refresh_token() but returns a TokenResult.
    """
    url = f"{host.rstrip('/')}/cloudide/api/v3/trae/oauth/ExchangeToken"
    payload = {
        "ClientID": client_id,
        "RefreshToken": refresh_token,
        "ClientSecret": "-",
        "UserID": user_id,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        text = resp.text

    if resp.status_code != 200:
        raise RuntimeError(f"ExchangeToken HTTP [{resp.status_code}]: {text[:500]}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"ExchangeToken: invalid JSON: {text[:500]}")

    err = data.get("ResponseMetadata", {}).get("Error")
    if err:
        return TokenResult(error_code=err.get("Code", ""), error_message=err.get("Message", ""))

    result = data.get("Result") or data.get("result") or data
    tr = TokenResult(
        token=result.get("Token") or result.get("token") or "",
        refresh_token=result.get("RefreshToken") or result.get("refreshToken") or refresh_token,
        user_id=result.get("UserID") or result.get("userId") or user_id,
        tenant_id=result.get("TenantID") or "",
        expires_at=result.get("TokenExpireAt") or result.get("tokenExpireAt") or 0,
        refresh_expires_at=result.get("RefreshExpireAt") or result.get("refreshExpireAt") or 0,
    )
    return tr
