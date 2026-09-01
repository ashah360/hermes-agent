"""Exact-ratio PCM geometry conversion for the realtime lane.

Discord voice: 48 kHz, stereo, s16le (3840 bytes / 20 ms frame).
gpt-realtime: 24 kHz, mono, pcm16.

The 2:1 integer ratio keeps this deterministic and cheap (numpy vector ops,
no resampler dependency): downsample averages adjacent sample pairs (mild
anti-aliasing), upsample linearly interpolates. numpy ships in the optional
"voice" extra — already a hard requirement of the mixer path.
"""

from __future__ import annotations


def _np():
    import numpy as np  # noqa: PLC0415 — lazy: optional voice extra

    return np


def discord_pcm_to_realtime(pcm_48k_stereo: bytes) -> bytes:
    """48 kHz stereo s16le → 24 kHz mono pcm16 (byte count / 4)."""
    if not pcm_48k_stereo:
        return b""
    np = _np()
    samples = np.frombuffer(pcm_48k_stereo, dtype=np.int16)
    if len(samples) % 2:
        samples = samples[:-1]
    # Stereo downmix: average L/R.
    mono = samples.reshape(-1, 2).mean(axis=1)
    # 2:1 decimation with pair averaging (cheap anti-alias).
    if len(mono) % 2:
        mono = mono[:-1]
    down = mono.reshape(-1, 2).mean(axis=1)
    return np.clip(np.round(down), -32768, 32767).astype(np.int16).tobytes()


def realtime_pcm_to_discord(pcm_24k_mono: bytes) -> bytes:
    """24 kHz mono pcm16 → 48 kHz stereo s16le (byte count * 4)."""
    if not pcm_24k_mono:
        return b""
    np = _np()
    mono = np.frombuffer(pcm_24k_mono, dtype=np.int16).astype(np.float32)
    if len(mono) == 0:
        return b""
    # 1:2 upsample: linear interpolation between neighbours.
    mid = (mono + np.append(mono[1:], mono[-1])) / 2.0
    up = np.empty(len(mono) * 2, dtype=np.float32)
    up[0::2] = mono
    up[1::2] = mid
    up16 = np.clip(np.round(up), -32768, 32767).astype(np.int16)
    stereo = np.repeat(up16[:, None], 2, axis=1).reshape(-1)
    return stereo.tobytes()
