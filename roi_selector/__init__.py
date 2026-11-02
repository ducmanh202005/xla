"""Local Streamlit component, no CDN or frontend build step."""

from pathlib import Path
import streamlit.components.v1 as components
from images import preview_url

_component = components.declare_component('m3diff_roi', path=str(Path(__file__).parent / 'frontend'))


def select_roi(image, key, value=None, revision=0):
    return _component(image=preview_url(image), value=value, revision=revision,
                      key=key, default={'roi': value, 'revision': revision})
