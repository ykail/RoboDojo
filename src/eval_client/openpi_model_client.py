"""Client adapter for OpenPI's native MsgPack WebSocket inference protocol."""

import time
from typing import Any

import msgpack
import numpy as np

_CAMERA_MAP = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
_STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
_STATE_DIMS = (6, 1, 6, 1)


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" not in value:
        return value
    return np.ndarray(
        buffer=value[b"data"],
        dtype=np.dtype(value[b"dtype"]),
        shape=value[b"shape"],
    )


def _as_chw_uint8(image: Any, *, camera_name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Camera {camera_name!r} must be an RGB image, got {array.shape}")
    if array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    elif array.shape[0] != 3:
        raise ValueError(f"Camera {camera_name!r} must have three RGB channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Camera {camera_name!r} contains non-finite values")
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(array, dtype=np.uint8)


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert a RoboDojo observation into OpenPI's ``images/state/prompt`` payload."""
    vision = observation.get("vision")
    state = observation.get("state")
    if not isinstance(vision, dict):
        raise KeyError("Observation is missing its vision mapping")
    if not isinstance(state, dict):
        raise KeyError("Observation is missing its state mapping")

    images = {}
    for robo_name, openpi_name in _CAMERA_MAP.items():
        try:
            images[openpi_name] = _as_chw_uint8(vision[robo_name]["color"], camera_name=robo_name)
        except KeyError as exc:
            raise KeyError(f"Observation is missing camera {robo_name!r}") from exc

    state_parts = []
    for key, dimension in zip(_STATE_KEYS, _STATE_DIMS, strict=True):
        if key not in state:
            raise KeyError(f"Observation is missing state field {key!r}")
        part = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if part.size != dimension:
            raise ValueError(f"State field {key!r} must contain {dimension} values, got {part.size}")
        state_parts.append(part)
    packed_state = np.concatenate(state_parts)
    if not np.all(np.isfinite(packed_state)):
        raise ValueError("Observation state contains non-finite values")

    payload: dict[str, Any] = {"images": images, "state": packed_state}
    instruction = observation.get("instruction")
    if isinstance(instruction, str) and instruction:
        payload["prompt"] = instruction
    return payload


def decode_actions(response: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    if "actions" not in response:
        raise KeyError("OpenPI inference response is missing actions")
    chunk = np.asarray(response["actions"], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 14:
        raise ValueError(f"OpenPI actions must have shape [H, 14] with H > 0, got {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("OpenPI actions contain non-finite values")
    return [
        {
            "left_arm_joint_state": step[0:6].copy(),
            "left_ee_joint_state": step[6:7].copy(),
            "right_arm_joint_state": step[7:13].copy(),
            "right_ee_joint_state": step[13:14].copy(),
        }
        for step in chunk
    ]


class OpenPiModelClient:
    """Expose native OpenPI inference through the XPolicyLab model-client API."""

    def __init__(
        self,
        *,
        url: str,
        connect_timeout_s: float = 30.0,
        max_connect_attempts: int = 10,
        connect_retry_delay_s: float = 5.0,
    ):
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.max_connect_attempts = max_connect_attempts
        self.connect_retry_delay_s = connect_retry_delay_s
        self._ws: Any | None = None
        self._latest_observations: list[dict[str, Any]] | None = None
        self.metadata: dict[str, Any] = {}

    def _connect(self) -> None:
        import websockets.sync.client

        last_error: Exception | None = None
        for attempt in range(1, self.max_connect_attempts + 1):
            try:
                self._ws = websockets.sync.client.connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    open_timeout=self.connect_timeout_s,
                )
                metadata = self._ws.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"OpenPI server returned an error during connection: {metadata}")
                decoded = msgpack.unpackb(metadata, object_hook=_unpack_numpy)
                if not isinstance(decoded, dict):
                    raise RuntimeError("OpenPI server metadata must be a MsgPack map")
                self.metadata = decoded
                return
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt < self.max_connect_attempts:
                    time.sleep(self.connect_retry_delay_s)
        raise ConnectionError(f"Could not connect to OpenPI server at {self.url}") from last_error

    def _infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(2):
            try:
                if self._ws is None:
                    self._connect()
                assert self._ws is not None
                self._ws.send(msgpack.packb(observation, default=_pack_numpy))
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"OpenPI inference error: {response}")
                decoded = msgpack.unpackb(response, object_hook=_unpack_numpy)
                if not isinstance(decoded, dict):
                    raise RuntimeError("OpenPI inference response must be a MsgPack map")
                return decoded
            except Exception:
                self.close()
                if attempt:
                    raise
        raise AssertionError("Unreachable")

    def call(self, func_name: str | None = None, obs: Any = None, **_: Any) -> Any:
        if func_name == "reset":
            self._latest_observations = None
            return None
        if func_name == "update_obs":
            self._latest_observations = [encode_observation(obs)]
            return None
        if func_name == "update_obs_batch":
            self._latest_observations = [encode_observation(item) for item in obs]
            return None
        if func_name == "get_action":
            if self._latest_observations is None:
                raise RuntimeError("update_obs must be called before get_action")
            return decode_actions(self._infer(self._latest_observations[0]))
        if func_name == "get_action_batch":
            if self._latest_observations is None:
                raise RuntimeError("update_obs_batch must be called before get_action_batch")
            return [decode_actions(self._infer(observation)) for observation in self._latest_observations]
        if func_name == "trial_end":
            return None
        raise NotImplementedError(f"Unsupported OpenPI model call: {func_name}")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
