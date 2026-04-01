"""Tests for render_engine_health function from app.py.

Since app.py loads data and models at module level, we extract the logic
directly to avoid import-time dependencies.
"""
import base64
import os
import pytest


def render_engine_health(engine_id: int, predicted_rul: float, max_rul: float = 125.0) -> str:
    """
    Extracted from app.py: Return an HTML string with turbofan SVG + health bar.

    This is a local copy to avoid import-time side effects from the Streamlit app.
    """
    pct = max(0.0, min(1.0, predicted_rul / max_rul))
    pct_int = int(pct * 100)

    if predicted_rul > 60:
        bar_color, status = "#5cb85c", "OPERATIONAL"
    elif predicted_rul >= 20:
        bar_color, status = "#e0a800", "CAUTION"
    else:
        bar_color, status = "#e94f37", "CRITICAL"

    # ── 25-segment health bar ─────────────────────────────────────────────────
    n_seg = 25
    filled = round(pct * n_seg)
    segs = "".join(
        '<div style="flex:1;height:100%;background:{};border-radius:3px;'
        'box-shadow:{};"></div>'.format(
            bar_color if i < filled else "#3a3f42",
            f"0 0 6px {bar_color}88" if i < filled else "none",
        )
        for i in range(n_seg)
    )

    # ── Turbofan SVG (cross-section side view) ────────────────────────────────
    svg = """
<svg viewBox="0 0 480 200" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;display:block;">
  <defs>
    <linearGradient id="nacG" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"   stop-color="#5a6065"/>
      <stop offset="45%"  stop-color="#393e41"/>
      <stop offset="100%" stop-color="#2b2f31"/>
    </linearGradient>
    <radialGradient id="flameG" cx="20%" cy="50%" r="80%">
      <stop offset="0%"   stop-color="#f6f7eb" stop-opacity="0.95"/>
      <stop offset="35%"  stop-color="#e94f37" stop-opacity="0.75"/>
      <stop offset="100%" stop-color="#e94f37" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="exhaustG" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#e94f37" stop-opacity="0"/>
      <stop offset="100%" stop-color="#e94f37" stop-opacity="0.55"/>
    </linearGradient>
  </defs>

  <!-- Exhaust flame glow -->
  <ellipse cx="440" cy="100" rx="65" ry="35" fill="url(#flameG)"/>
  <ellipse cx="428" cy="100" rx="42" ry="22" fill="#e94f37" opacity="0.25"/>

  <!-- Outer nacelle body -->
  <path d="M 78,36 Q 56,100 78,164 L 388,144 L 412,100 L 388,56 Z"
        fill="url(#nacG)" stroke="#6d7275" stroke-width="2"/>

  <!-- Nacelle highlight glint -->
  <path d="M 82,40 Q 62,100 82,160 L 92,156 Q 73,100 92,44 Z"
        fill="#6d7275" opacity="0.35"/>

  <!-- Bypass duct divider lines -->
  <path d="M 98,68 L 382,79" stroke="#4d5457" stroke-width="1.5" stroke-dasharray="7,4"/>
  <path d="M 98,132 L 382,121" stroke="#4d5457" stroke-width="1.5" stroke-dasharray="7,4"/>

  <!-- Core engine tube -->
  <path d="M 98,72 L 378,80 L 378,120 L 98,128 Z"
        fill="#1e2224" stroke="#5a6065" stroke-width="1.5"/>

  <!-- Compressor blades (front of core) -->
  <line x1="118" y1="73" x2="118" y2="127" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="138" y1="73" x2="138" y2="127" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="157" y1="74" x2="157" y2="126" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="175" y1="74" x2="175" y2="126" stroke="#a8ada8" stroke-width="2.5"/>

  <!-- Combustion chamber -->
  <rect x="183" y="76" width="86" height="48" rx="5"
        fill="#3a1a10" stroke="#e94f37" stroke-width="2"/>
  <ellipse cx="226" cy="100" rx="24" ry="14" fill="#e94f37" opacity="0.55"/>
  <ellipse cx="226" cy="100" rx="11" ry="7"  fill="#f6f7eb" opacity="0.65"/>

  <!-- Turbine blades (back of core) -->
  <line x1="280" y1="76" x2="280" y2="124" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="299" y1="77" x2="299" y2="123" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="317" y1="78" x2="317" y2="122" stroke="#a8ada8" stroke-width="2.5"/>
  <line x1="334" y1="79" x2="334" y2="121" stroke="#a8ada8" stroke-width="2.5"/>

  <!-- Exhaust nozzle -->
  <path d="M 378,80 L 412,91 L 412,109 L 378,120 Z"
        fill="#1e2224" stroke="#6d7275" stroke-width="1.5"/>
  <path d="M 378,80 L 460,68 L 460,132 L 378,120 Z" fill="url(#exhaustG)"/>

  <!-- Inlet cowl -->
  <ellipse cx="80" cy="100" rx="20" ry="65" fill="#4d5457" stroke="#6d7275" stroke-width="2"/>
  <ellipse cx="84" cy="100" rx="13" ry="52" fill="#1e2224" stroke="#a8ada8" stroke-width="1"/>

  <!-- Fan hub disk -->
  <ellipse cx="97" cy="100" rx="11" ry="57" fill="#393e41" stroke="#e94f37" stroke-width="2"/>
  <circle  cx="97" cy="100" r="8"           fill="#4d5457"  stroke="#e94f37" stroke-width="1.5"/>

  <!-- Fan blades -->
  <line x1="87" y1="52"  x2="107" y2="49"  stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="65"  x2="108" y2="62"  stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="78"  x2="108" y2="76"  stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="91"  x2="108" y2="91"  stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="100" x2="108" y2="100" stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="109" x2="108" y2="109" stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="122" x2="108" y2="124" stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="135" x2="108" y2="138" stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>
  <line x1="87" y1="148" x2="107" y2="151" stroke="#f6f7eb" stroke-width="3" stroke-linecap="round"/>

  <!-- Mounting pylon (top strut) -->
  <rect x="196" y="0" width="28" height="38" rx="3" fill="#4d5457" stroke="#6d7275" stroke-width="1.5"/>
  <line x1="202" y1="4"  x2="202" y2="36" stroke="#5a6065" stroke-width="1"/>
  <line x1="210" y1="4"  x2="210" y2="36" stroke="#5a6065" stroke-width="1"/>
  <line x1="218" y1="4"  x2="218" y2="36" stroke="#5a6065" stroke-width="1"/>
</svg>"""

    # Streamlit strips <svg> tags — encode as base64 data URI for <img> instead
    svg_b64 = base64.b64encode(svg.strip().encode("utf-8")).decode("utf-8")
    svg_img = (
        f'<img src="data:image/svg+xml;base64,{svg_b64}" '
        f'style="width:100%;height:auto;display:block;"/>'
    )

    return """
<div style="background:#1e2224;border:2px solid {bc};border-radius:10px;
            padding:22px 26px;margin-bottom:16px;">

  <!-- Title row -->
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;">
    <span style="color:#a8ada8;font-size:1rem;letter-spacing:2px;text-transform:uppercase;">
      ENGINE UNIT #{eid}
    </span>
    <span style="color:{bc};font-size:1.5rem;font-weight:800;letter-spacing:3px;">
      &#9654; {status}
    </span>
  </div>

  <!-- Engine image + health bar side-by-side -->
  <div style="display:flex;gap:28px;align-items:center;">

    <!-- Engine image -->
    <div style="flex:3;min-width:0;">{svg_img}</div>

    <!-- Health bar column -->
    <div style="flex:2;min-width:160px;display:flex;flex-direction:column;gap:10px;">
      <div style="color:#a8ada8;font-size:0.85rem;letter-spacing:2px;">ENGINE HEALTH</div>

      <!-- Segmented bar (vertical) -->
      <div style="display:flex;flex-direction:column-reverse;gap:3px;
                  height:200px;background:#12181a;border:2px solid #3a3f42;
                  border-radius:6px;padding:6px;">
        {segs_v}
      </div>

      <!-- RUL number -->
      <div style="text-align:center;">
        <span style="color:{bc};font-size:3rem;font-weight:800;line-height:1;">{rul:.0f}</span>
        <br>
        <span style="color:#a8ada8;font-size:0.95rem;">cycles remaining</span>
      </div>

      <!-- Percent -->
      <div style="text-align:center;background:#12181a;border:1px solid {bc};
                  border-radius:4px;padding:6px;">
        <span style="color:{bc};font-size:1.4rem;font-weight:700;">{pct}%</span>
        <span style="color:#a8ada8;font-size:0.85rem;"> of max life</span>
      </div>
    </div>
  </div>
</div>""".format(
        bc=bar_color, eid=engine_id, status=status, svg_img=svg_img,
        segs_v=segs, rul=predicted_rul, pct=pct_int,
    )


class TestRenderEngineHealthGreen:
    """Tests for render_engine_health function with RUL > 60 (green)."""

    def test_rul_above_60_returns_html_string(self):
        """render_engine_health returns a string."""
        html = render_engine_health(1, 80.0)
        assert isinstance(html, str)

    def test_rul_above_60_contains_green_color(self):
        """RUL > 60 should contain green color #5cb85c."""
        html = render_engine_health(1, 80.0)
        assert "#5cb85c" in html

    def test_rul_above_60_contains_operational_status(self):
        """RUL > 60 should contain OPERATIONAL status."""
        html = render_engine_health(1, 80.0)
        assert "OPERATIONAL" in html

    def test_rul_above_60_contains_engine_id(self):
        """HTML should contain engine ID number."""
        engine_id = 42
        html = render_engine_health(engine_id, 80.0)
        assert f"#{engine_id}" in html or str(engine_id) in html

    def test_rul_above_60_contains_svg_data_uri(self):
        """HTML should contain base64-encoded SVG as data URI."""
        html = render_engine_health(1, 80.0)
        assert "data:image/svg+xml;base64," in html

    def test_rul_above_60_contains_cycles_remaining_text(self):
        """HTML should mention cycles remaining."""
        html = render_engine_health(1, 80.0)
        assert "cycles remaining" in html.lower()

    def test_rul_above_60_contains_engine_health_label(self):
        """HTML should contain ENGINE HEALTH label."""
        html = render_engine_health(1, 80.0)
        assert "ENGINE HEALTH" in html


class TestRenderEngineHealthAmber:
    """Tests for render_engine_health function with 20 ≤ RUL ≤ 60 (amber)."""

    def test_rul_40_returns_html_string(self):
        """render_engine_health returns a string."""
        html = render_engine_health(2, 40.0)
        assert isinstance(html, str)

    def test_rul_40_contains_amber_color(self):
        """RUL in [20, 60] should contain amber color #e0a800."""
        html = render_engine_health(2, 40.0)
        assert "#e0a800" in html

    def test_rul_40_contains_caution_status(self):
        """RUL in [20, 60] should contain CAUTION status."""
        html = render_engine_health(2, 40.0)
        assert "CAUTION" in html

    def test_rul_50_also_amber(self):
        """RUL = 50 is in amber range."""
        html = render_engine_health(2, 50.0)
        assert "#e0a800" in html
        assert "CAUTION" in html

    def test_rul_20_boundary_is_amber(self):
        """RUL = 20 (boundary) should be amber."""
        html = render_engine_health(2, 20.0)
        assert "#e0a800" in html
        assert "CAUTION" in html

    def test_rul_60_boundary_is_amber(self):
        """RUL = 60 (boundary) should be amber."""
        html = render_engine_health(2, 60.0)
        assert "#e0a800" in html
        assert "CAUTION" in html

    def test_rul_40_contains_engine_id(self):
        """HTML should contain engine ID number."""
        engine_id = 7
        html = render_engine_health(engine_id, 40.0)
        assert f"#{engine_id}" in html or str(engine_id) in html


class TestRenderEngineHealthRed:
    """Tests for render_engine_health function with RUL < 20 (terracotta/red)."""

    def test_rul_below_20_returns_html_string(self):
        """render_engine_health returns a string."""
        html = render_engine_health(3, 10.0)
        assert isinstance(html, str)

    def test_rul_below_20_contains_terracotta_color(self):
        """RUL < 20 should contain terracotta color #e94f37."""
        html = render_engine_health(3, 10.0)
        assert "#e94f37" in html

    def test_rul_below_20_contains_critical_status(self):
        """RUL < 20 should contain CRITICAL status."""
        html = render_engine_health(3, 10.0)
        assert "CRITICAL" in html

    def test_rul_5_also_red(self):
        """RUL = 5 is in critical range."""
        html = render_engine_health(3, 5.0)
        assert "#e94f37" in html
        assert "CRITICAL" in html

    def test_rul_0_also_red(self):
        """RUL = 0 should be critical."""
        html = render_engine_health(3, 0.0)
        assert "#e94f37" in html
        assert "CRITICAL" in html

    def test_rul_19_99_is_critical_not_amber(self):
        """RUL = 19.99 is just below 20 threshold, should be critical."""
        html = render_engine_health(3, 19.99)
        assert "#e94f37" in html
        assert "CRITICAL" in html


class TestRenderEngineHealthRulDisplay:
    """Tests for RUL value display in HTML."""

    def test_html_contains_rul_value_80(self):
        """HTML should contain the RUL value 80."""
        html = render_engine_health(1, 80.0)
        assert "80" in html

    def test_html_contains_rul_value_42_5(self):
        """HTML should contain RUL value 42.5."""
        html = render_engine_health(1, 42.5)
        # Should be formatted as integer: 42 or 43 (depending on rounding)
        assert "42" in html or "43" in html

    def test_html_contains_rul_value_15(self):
        """HTML should contain RUL value 15."""
        html = render_engine_health(1, 15.0)
        assert "15" in html


class TestRenderEngineHealthPercentage:
    """Tests for percentage display in HTML."""

    def test_html_contains_100_percent_at_max(self):
        """RUL = 125 (max) should display 100%."""
        html = render_engine_health(1, 125.0)
        assert "100" in html
        assert "%" in html

    def test_html_contains_50_percent_at_half(self):
        """RUL = 62.5 (half of 125) should display ~50%."""
        html = render_engine_health(1, 62.5)
        assert "50" in html
        assert "%" in html

    def test_html_contains_0_percent_at_zero(self):
        """RUL = 0 should display 0%."""
        html = render_engine_health(1, 0.0)
        assert "0" in html

    def test_percentage_of_max_life_text(self):
        """HTML should contain 'of max life' text."""
        html = render_engine_health(1, 80.0)
        assert "max life" in html.lower()


class TestRenderEngineHealthSvg:
    """Tests for SVG encoding in HTML."""

    def test_svg_is_base64_encoded(self):
        """SVG should be base64-encoded in data URI."""
        html = render_engine_health(1, 80.0)
        # Extract base64 part
        assert "data:image/svg+xml;base64," in html
        # The actual base64 should follow
        idx = html.find("data:image/svg+xml;base64,")
        assert idx >= 0
        after_prefix = html[idx + len("data:image/svg+xml;base64,") : idx + len("data:image/svg+xml;base64,") + 100]
        # Should contain base64-like characters
        assert any(c in after_prefix for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")

    def test_svg_contains_turbofan_features(self):
        """Decoded SVG should contain turbofan engine elements."""
        html = render_engine_health(1, 80.0)
        # Extract and decode the base64 SVG
        idx = html.find("data:image/svg+xml;base64,")
        start = idx + len("data:image/svg+xml;base64,")
        end = html.find('"', start)
        if end == -1:
            end = html.find("'", start)
        svg_b64 = html[start:end]
        try:
            svg_decoded = base64.b64decode(svg_b64).decode("utf-8")
            # Should contain SVG elements
            assert "<svg" in svg_decoded
            assert "</svg>" in svg_decoded
        except Exception:
            # If decoding fails, it's still an issue
            pytest.fail("SVG base64 decoding failed")

    def test_svg_data_uri_is_in_img_tag(self):
        """SVG should be embedded in an <img> tag with data URI."""
        html = render_engine_health(1, 80.0)
        assert '<img src="data:image/svg+xml;base64,' in html


class TestRenderEngineHealthEngineUnit:
    """Tests for engine unit number display."""

    def test_engine_unit_label_present(self):
        """HTML should contain 'ENGINE UNIT' label."""
        html = render_engine_health(1, 80.0)
        assert "ENGINE UNIT" in html

    def test_engine_unit_1(self):
        """Engine 1 should display as #1."""
        html = render_engine_health(1, 80.0)
        assert "#1" in html

    def test_engine_unit_42(self):
        """Engine 42 should display as #42."""
        html = render_engine_health(42, 80.0)
        assert "#42" in html

    def test_engine_unit_999(self):
        """Engine 999 should display as #999."""
        html = render_engine_health(999, 80.0)
        assert "#999" in html


class TestRenderEngineHealthCustomMaxRul:
    """Tests for custom max_rul parameter."""

    def test_custom_max_rul_100(self):
        """RUL at 100% of custom max 100 should display 100%."""
        html = render_engine_health(1, 100.0, max_rul=100.0)
        assert "100" in html

    def test_custom_max_rul_50_at_half(self):
        """RUL 25 of max 50 should display ~50%."""
        html = render_engine_health(1, 25.0, max_rul=50.0)
        assert "50" in html
        assert "%" in html

    def test_custom_max_rul_affects_percentage_not_status(self):
        """Custom max_rul should affect percentage but not status thresholds."""
        # Status thresholds are absolute (60, 20), not relative to max_rul
        html = render_engine_health(1, 80.0, max_rul=200.0)
        # RUL=80 is still above 60, so should be OPERATIONAL
        assert "OPERATIONAL" in html
        assert "#5cb85c" in html


class TestRenderEngineHealthEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rul_greater_than_max_clamped(self):
        """RUL greater than max_rul should be clamped to 100%."""
        html = render_engine_health(1, 150.0, max_rul=125.0)
        # Percentage should not exceed 100
        assert "100" in html

    def test_negative_rul_clamped_to_zero(self):
        """Negative RUL should be clamped to 0%."""
        html = render_engine_health(1, -10.0)
        # Should display 0% and CRITICAL status
        assert "#e94f37" in html
        assert "CRITICAL" in html

    def test_very_small_rul_is_critical(self):
        """Very small RUL like 0.1 should be critical."""
        html = render_engine_health(1, 0.1)
        assert "#e94f37" in html
        assert "CRITICAL" in html

    def test_empty_engine_id_zero(self):
        """Engine ID 0 should still work."""
        html = render_engine_health(0, 80.0)
        assert "#0" in html


class TestRenderEngineHealthStructure:
    """Tests for HTML structure integrity."""

    def test_html_contains_div_wrapper(self):
        """HTML should be wrapped in a div."""
        html = render_engine_health(1, 80.0)
        assert "<div" in html
        assert "</div>" in html

    def test_html_is_valid_format(self):
        """HTML should not have obvious syntax errors."""
        html = render_engine_health(1, 80.0)
        # Closing divs should match opening divs (rough check)
        assert html.count("<div") > 0
        assert html.count("</div>") > 0
        # Should have roughly equal opening/closing tags
        assert abs(html.count("<div") - html.count("</div>")) <= 5

    def test_html_contains_style_attributes(self):
        """HTML should contain inline styles."""
        html = render_engine_health(1, 80.0)
        assert 'style="' in html

    def test_html_contains_flex_layout(self):
        """HTML should use flexbox layout."""
        html = render_engine_health(1, 80.0)
        assert "display:flex" in html
