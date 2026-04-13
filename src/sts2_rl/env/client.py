from __future__ import annotations

from typing import Any

import httpx

from .models import (
    ActionRequest,
    ActionResponsePayload,
    ApiEnvelope,
    AvailableActionsPayload,
    GameStatePayload,
    HealthPayload,
)


class Sts2ApiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.retryable = retryable


class Sts2Client:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("Provide either client or transport, not both.")

        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Sts2Client:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def get_health(self) -> HealthPayload:
        payload = self._request("GET", "/health")
        return HealthPayload.model_validate(payload.data or {})

    def get_state(self) -> GameStatePayload:
        payload = self._request("GET", "/state")
        return GameStatePayload.model_validate(payload.data or {})

    def get_available_actions(self) -> AvailableActionsPayload:
        payload = self._request("GET", "/actions/available")
        return AvailableActionsPayload.model_validate(payload.data or {})

    def post_action(self, request: ActionRequest) -> ActionResponsePayload:
        payload = self._request(
            "POST",
            "/action",
            json=request.model_dump(exclude_none=True),
        )
        return ActionResponsePayload.model_validate(payload.data or {})

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiEnvelope:
        response = self._client.request(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError(f"Non-JSON response from STS2-Agent at {path}") from exc

        envelope = ApiEnvelope.model_validate(body)
        if not envelope.ok:
            error = envelope.error
            raise Sts2ApiError(
                status_code=response.status_code,
                code=error.code if error is not None else "unknown_error",
                message=error.message if error is not None else "Unknown STS2-Agent error.",
                details=error.details if error is not None else None,
                retryable=error.retryable if error is not None else False,
            )

        response.raise_for_status()
        return envelope
