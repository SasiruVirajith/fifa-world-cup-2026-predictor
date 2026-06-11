"""macOS-styled sidebar controls (toggle + slider) for Streamlit."""

import os

import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")
_component = components.declare_component("macos_sidebar_controls", path=_FRONTEND)


def macos_sidebar_controls(*, dark_mode: bool, top_n: int, key=None):
    """Return (dark_mode, top_n) from custom macOS controls."""
    result = _component(dark_mode=dark_mode, top_n=top_n, key=key, default=None)
    if not result:
        return dark_mode, top_n
    return bool(result.get("dark_mode", dark_mode)), int(result.get("top_n", top_n))
