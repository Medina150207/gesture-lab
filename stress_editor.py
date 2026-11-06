"""Live photo editor. Inference receives the PNG displayed by the browser."""
import base64
import io
from pathlib import Path

from PIL import Image, ImageOps
import streamlit as st


HTML = """
<div class="photo"><canvas aria-label="Adjusted photo"></canvas></div>
<label>Brightness <output id="light-value">1.0×</output>
  <input id="light" type="range" min="0.2" max="1.8" step="0.1" value="1" disabled></label>
<label>Blur <output id="blur-value">0 px</output>
  <input id="blur" type="range" min="0" max="8" step="0.5" value="0" disabled></label>
<div class="actions"><button id="test" disabled>Run test</button><button id="reset" disabled>Reset</button></div>
<span role="status"></span>
"""
CSS = """
.photo {background:#f7eff7;border-radius:16px;text-align:center;line-height:0}
canvas {width:auto;height:auto;max-width:100%;max-height:310px;border-radius:12px}
label {display:block;margin:12px 0;color:#542747;font:600 15px system-ui}
output {float:right;font-variant-numeric:tabular-nums}
input {display:block;width:100%;margin:5px 0;accent-color:#db428f;cursor:pointer}
.actions {display:flex;gap:10px} button {border:1px solid #c484a9;background:white;
 color:#702b56;border-radius:20px;padding:8px 20px;cursor:pointer;font:600 14px system-ui}
#test {background:#8e3567;color:white} button:disabled {opacity:.5}
"""
widget = st.components.v2.component(
    "gesture_stress_editor", html=HTML, css=CSS,
    js=Path(__file__).with_name("stress_editor.js").read_text(encoding="utf-8"),
)


def render_editor(image, identity):
    preview = ImageOps.contain(image, (480, 480))
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return widget(
        key=f"editor_{identity}", data={"photo": base64.b64encode(buffer.getvalue()).decode("ascii")},
        default={"settings": [1.0, 0.0]}, on_settings_change=lambda: None,
        on_test_change=lambda: None, on_reset_change=lambda: None,
    )


def tested_photo(payload):
    encoded = payload["pixels"]
    if len(encoded) > 1_500_000:
        raise ValueError("Preview exceeds the size limit.")
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as photo:
        if photo.format != "PNG" or max(photo.size) > 480:
            raise ValueError("Invalid preview.")
        return photo.convert("RGB"), payload["settings"]
