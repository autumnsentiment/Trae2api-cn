"""Enterprise authorization state machine for Trae CN.

This module models the enterprise authorization flow without storing or
exporting real credentials.  It is intentionally independent from the Trae
Work account pool: an enterprise login must never fall back to a personal or
Work JWT.

The endpoint names below are recovered from the installed Trae CN client.
Payloads are represented as typed dictionaries so an operator can swap in a
controlled test double or a customer-owned integration.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol


class EnterpriseAuthState(str, Enum):
    SIGNED_OUT = "signed_out"
    AUTH_PENDING = "auth_pending"
    TOKEN_RECEIVED = "token_received"
    TENANT_RESOLVED = "tenant_resolved"
    ACCOUNT_READY = "account_ready"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True)
class EnterpriseIdentity:
    """Non-secret identity fields used to bind a session to one tenant."""

    user_id: str
    tenant_id: str
    scope: str = ""
    region: str = ""
    organization_slug: str = ""


@dataclass
class EnterpriseTokens:
    """Credential container.

    The values must stay in memory or in a caller-provided secret store.
    They must not be written to logs, reports, or source files.
    """

    access_token: str = field(repr=False, default="")
    refresh_token: str = field(repr=False, default="")
    token_type: str = "Bearer"
    expires_at: float = 0.0
    refresh_expires_at: float = 0.0

    def is_access_valid(self, skew_seconds: int = 300) -> bool:
        return bool(self.access_token) and (
            self.expires_at <= 0 or self.expires_at - time.time() > skew_seconds
        )

    def needs_refresh(self, skew_seconds: int = 300) -> bool:
        return bool(self.refresh_token) and not self.is_access_valid(skew_seconds)


@dataclass
class EnterpriseAuthContext:
    state: EnterpriseAuthState = EnterpriseAuthState.SIGNED_OUT
    identity: Optional[EnterpriseIdentity] = None
    tokens: EnterpriseTokens = field(default_factory=EnterpriseTokens)
    provider_specific: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    revision: int = 0

    def credential_fingerprint(self) -> str:
        """Return a non-reversible correlation id for diagnostics only."""

        if not self.tokens.access_token:
            return ""
        return hashlib.sha256(
            self.tokens.access_token.encode("utf-8")
        ).hexdigest()[:12]


class EnterpriseAuthTransport(Protocol):
    """Adapter boundary for login, user lookup, and token refresh."""

    async def exchange_authorization_code(
        self, code: str, redirect_uri: str
    ) -> Mapping[str, Any]:
        ...

    async def fetch_user_info(self, access_token: str) -> Mapping[str, Any]:
        ...

    async def refresh_access_token(self, refresh_token: str) -> Mapping[str, Any]:
        ...


class EnterpriseAuthFlow:
    """Deterministic state machine for the enterprise login chain.

    Typical sequence:

        SIGNED_OUT
          -> AUTH_PENDING
          -> TOKEN_RECEIVED
          -> TENANT_RESOLVED
          -> ACCOUNT_READY

    A token alone is not sufficient: the account is not ready until a tenant
    id is resolved.  This prevents an enterprise context from silently using
    another tenant or a personal account.
    """

    def __init__(self, transport: EnterpriseAuthTransport):
        self._transport = transport
        self.context = EnterpriseAuthContext()

    def _advance(
        self,
        state: EnterpriseAuthState,
        *,
        error_code: str = "",
    ) -> EnterpriseAuthContext:
        if self.context.state is not state or self.context.error_code != error_code:
            self.context.revision += 1
        self.context.state = state
        self.context.error_code = error_code
        return self.context

    async def begin(self) -> EnterpriseAuthContext:
        self.context = EnterpriseAuthContext(
            state=EnterpriseAuthState.AUTH_PENDING,
            revision=self.context.revision + 1,
        )
        return self.context

    async def complete_login(
        self,
        code: str,
        redirect_uri: str,
    ) -> EnterpriseAuthContext:
        """Exchange an authorization code and resolve the tenant.

        ``code`` may be supplied by a browser callback, a device flow, or an
        enterprise SSO adapter implemented by the caller.
        """

        if not code.strip():
            return self._advance(
                EnterpriseAuthState.FAILED,
                error_code="authorization_code_missing",
            )

        try:
            token_payload = await self._transport.exchange_authorization_code(
                code, redirect_uri
            )
            self.context.tokens = _tokens_from_payload(token_payload)
            self._advance(EnterpriseAuthState.TOKEN_RECEIVED)
        except Exception:
            return self._advance(
                EnterpriseAuthState.FAILED,
                error_code="token_exchange_failed",
            )

        if not self.context.tokens.access_token:
            return self._advance(
                EnterpriseAuthState.FAILED,
                error_code="access_token_missing",
            )

        try:
            profile = await self._transport.fetch_user_info(
                self.context.tokens.access_token
            )
            identity = _identity_from_profile(profile)
        except Exception:
            return self._advance(
                EnterpriseAuthState.FAILED,
                error_code="user_info_failed",
            )

        if not identity.tenant_id:
            return self._advance(
                EnterpriseAuthState.FAILED,
                error_code="tenant_missing",
            )

        self.context.identity = identity
        self._advance(EnterpriseAuthState.TENANT_RESOLVED)
        return self._advance(EnterpriseAuthState.ACCOUNT_READY)

    async def ensure_fresh_token(self) -> EnterpriseAuthContext:
        """Refresh only when the access token is near expiry."""

        if self.context.state is not EnterpriseAuthState.ACCOUNT_READY:
            return self.context
        if not self.context.tokens.needs_refresh():
            return self.context

        try:
            payload = await self._transport.refresh_access_token(
                self.context.tokens.refresh_token
            )
            refreshed = _tokens_from_payload(payload)
            if refreshed.access_token:
                refreshed.refresh_token = (
                    refreshed.refresh_token
                    or self.context.tokens.refresh_token
                )
                self.context.tokens = refreshed
                self.context.revision += 1
            else:
                self._advance(
                    EnterpriseAuthState.EXPIRED,
                    error_code="refresh_token_rejected",
                )
        except Exception:
            self._advance(
                EnterpriseAuthState.EXPIRED,
                error_code="refresh_failed",
            )
        return self.context

    def logout(self) -> EnterpriseAuthContext:
        self.context.tokens = EnterpriseTokens()
        self.context.identity = None
        self.context.provider_specific = {}
        return self._advance(EnterpriseAuthState.SIGNED_OUT)


def _tokens_from_payload(payload: Mapping[str, Any]) -> EnterpriseTokens:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    access = str(
        data.get("accessToken")
        or data.get("access_token")
        or data.get("token")
        or ""
    )
    refresh = str(
        data.get("refreshToken")
        or data.get("refresh_token")
        or ""
    )
    token_type = str(data.get("tokenType") or data.get("token_type") or "Bearer")
    expires_in = data.get("expiresIn") or data.get("expires_in") or 0
    try:
        expires_in = float(expires_in)
    except (TypeError, ValueError):
        expires_in = 0.0
    now = time.time()
    return EnterpriseTokens(
        access_token=access,
        refresh_token=refresh,
        token_type=token_type,
        expires_at=now + expires_in if expires_in > 0 else 0.0,
    )


def _identity_from_profile(payload: Mapping[str, Any]) -> EnterpriseIdentity:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    account = data.get("account") if isinstance(data.get("account"), Mapping) else {}
    organization = (
        data.get("organization")
        if isinstance(data.get("organization"), Mapping)
        else {}
    )
    return EnterpriseIdentity(
        user_id=str(
            data.get("userId")
            or data.get("user_id")
            or data.get("uid")
            or account.get("id")
            or ""
        ),
        tenant_id=str(
            data.get("tenantId")
            or data.get("tenant_id")
            or organization.get("tenant_id")
            or ""
        ),
        scope=str(data.get("scope") or ""),
        region=str(data.get("region") or data.get("loginRegion") or ""),
        organization_slug=str(
            organization.get("slug")
            or data.get("organizationSlug")
            or ""
        ),
    )


PRODUCT_TYPE_FREE = 0
PRODUCT_TYPE_PACKAGE = 2


@dataclass(frozen=True)
class EnterpriseAccountClassification:
    """Non-secret classification of one account against enterprise entitlements."""

    is_enterprise_candidate: bool
    has_tenant: bool
    has_package: bool
    has_free: bool
    tenant_fingerprint: str
    entitlements: tuple[int, ...]


def classify_enterprise_account(
    user_info: Mapping[str, Any],
    entitlement_usage: Mapping[str, Any],
) -> EnterpriseAccountClassification:
    """Classify an account without retaining raw tenant or credential values.

    Enterprise candidates have a tenant id and at least one Package
    entitlement.  A Package plus Free pair is the observed shape for tenant
    member accounts.  The tenant id is reduced to a stable fingerprint so
    callers can group members of the same tenant without exposing the id.
    """

    profile = (
        user_info.get("Result")
        if isinstance(user_info.get("Result"), Mapping)
        else user_info
    )
    tenant = str(
        profile.get("TenantID")
        or profile.get("TenantId")
        or profile.get("tenantId")
        or profile.get("tenant_id")
        or ""
    )
    packs = entitlement_usage.get("user_entitlement_pack_list")
    packs = packs if isinstance(packs, list) else []

    product_types: list[int] = []
    for pack in packs:
        if not isinstance(pack, Mapping):
            continue
        base = pack.get("entitlement_base_info")
        if not isinstance(base, Mapping):
            continue
        product_type = base.get("product_type")
        if isinstance(product_type, int) and not isinstance(product_type, bool):
            product_types.append(product_type)

    has_package = PRODUCT_TYPE_PACKAGE in product_types
    has_free = PRODUCT_TYPE_FREE in product_types
    return EnterpriseAccountClassification(
        is_enterprise_candidate=bool(tenant) and has_package,
        has_tenant=bool(tenant),
        has_package=has_package,
        has_free=has_free,
        tenant_fingerprint=(
            hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:12]
            if tenant
            else ""
        ),
        entitlements=tuple(product_types),
    )
