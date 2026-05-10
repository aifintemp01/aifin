"""PDF report generation service.

Synchronous — call via asyncio.run_in_executor from the queue worker.
"""
import io
import re
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from jinja2 import Environment, BaseLoader
from weasyprint import HTML as WeasyHTML


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(fig: plt.Figure) -> str:
    """Save matplotlib figure to base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return result


def _pretty_agent_name(agent_id: str) -> str:
    """'warren_buffett_agent_a1b2c3d4' -> 'Warren Buffett'"""
    name = re.sub(r"[_-][a-f0-9]{6,}.*$", "", agent_id, flags=re.IGNORECASE)
    name = re.sub(r"_agent$", "", name)
    return name.replace("_", " ").title()


def _pretty_reasoning(reasoning: Any, max_len: int = 350) -> str:
    """Normalise reasoning to a readable plain string."""
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        text = reasoning
    elif isinstance(reasoning, dict):
        parts = []
        for k, v in list(reasoning.items())[:8]:
            if isinstance(v, (str, int, float)) and v:
                parts.append(f"{k}: {v}")
        text = " | ".join(parts) if parts else str(reasoning)
    else:
        text = str(reasoning)
    return (text[:max_len] + "…") if len(text) > max_len else text


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

def _chart_signal_donut(ticker: str, analyst_signals: Dict) -> str:
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for agent_data in analyst_signals.values():
        sig = str(agent_data.get(ticker, {}).get("signal", "neutral")).lower()
        if sig in counts:
            counts[sig] += 1

    total = sum(counts.values())
    if total == 0:
        return ""

    data = [
        (counts["bullish"], "#22c55e", "Bullish"),
        (counts["bearish"], "#ef4444", "Bearish"),
        (counts["neutral"], "#f59e0b", "Neutral"),
    ]
    data = [(v, c, l) for v, c, l in data if v > 0]
    if not data:
        return ""
    sizes, colors, labels = zip(*data)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="white")
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0, f"{total}\nagents", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#111827")
    ax.legend(
        wedges, [f"{l} ({v})" for l, v in zip(labels, sizes)],
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=3, fontsize=7.5, frameon=False,
    )
    plt.tight_layout(pad=0.3)
    return _b64(fig)


def _chart_allocation_pie(decisions: Dict, current_prices: Dict) -> str:
    tickers = list(decisions.keys())
    if not tickers:
        return ""

    values = []
    for t in tickers:
        qty = float(decisions[t].get("quantity", 0))
        price = float(current_prices.get(t, 0))
        values.append(max(qty * price, 1.0))  # min 1 so every ticker appears

    colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
              "#f59e0b", "#ef4444", "#ec4899", "#64748b"]

    fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values, labels=tickers,
        colors=colors[: len(tickers)],
        autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    plt.tight_layout(pad=0.3)
    return _b64(fig)


def _chart_sparkline(ticker: str, prices: List) -> str:
    if not prices or len(prices) < 3:
        return ""

    closes = [float(p.close) for p in prices]
    color = "#22c55e" if closes[-1] >= closes[0] else "#ef4444"
    pct = (closes[-1] - closes[0]) / closes[0] * 100

    fig, ax = plt.subplots(figsize=(5.5, 2.2), facecolor="white")
    x = list(range(len(closes)))
    ax.plot(x, closes, color=color, linewidth=1.5, solid_capstyle="round")
    ax.fill_between(x, closes, min(closes), alpha=0.13, color=color)
    ax.set_xlim(0, len(closes) - 1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#e5e7eb")
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    sign = "+" if pct >= 0 else ""
    ax.set_title(f"{sign}{pct:.1f}% over period",
                 fontsize=8, color=color, loc="right", pad=3)
    plt.tight_layout(pad=0.3)
    return _b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-size:10.5pt;color:#1f2937;background:#fff;}
@page{margin:14mm 18mm 14mm 18mm;}
.hdr{background:#0f172a;color:#fff;padding:18px 22px;margin-bottom:22px;border-radius:6px;display:flex;justify-content:space-between;align-items:center;}
.hdr-brand{font-size:15pt;font-weight:700;letter-spacing:2px;}
.hdr-meta{font-size:8.5pt;color:#94a3b8;text-align:right;line-height:1.7;}
.hdr-meta .fn{font-size:10.5pt;color:#fff;font-weight:600;}
h1{font-size:13pt;font-weight:700;color:#0f172a;margin-bottom:10px;padding-bottom:5px;border-bottom:2px solid #e2e8f0;}
section{margin-bottom:24px;}
table{width:100%;border-collapse:collapse;font-size:9.5pt;}
th{background:#f1f5f9;color:#374151;font-weight:600;padding:7px 10px;text-align:left;border-bottom:1.5px solid #cbd5e1;}
td{padding:6px 10px;border-bottom:0.5px solid #e2e8f0;vertical-align:top;}
tr:last-child td{border-bottom:none;}
.badge{display:inline-block;padding:1.5px 7px;border-radius:10px;font-size:8pt;font-weight:600;}
.b-bullish{background:#dcfce7;color:#16a34a;}
.b-bearish{background:#fee2e2;color:#dc2626;}
.b-neutral{background:#fef3c7;color:#d97706;}
.b-buy{background:#dcfce7;color:#16a34a;}
.b-sell{background:#fee2e2;color:#dc2626;}
.b-hold{background:#fef3c7;color:#d97706;}
.b-short{background:#f3e8ff;color:#7c3aed;}
.b-cover{background:#dbeafe;color:#2563eb;}
.ts{page-break-inside:avoid;border-left:3.5px solid #3b82f6;background:#f8fafc;padding:14px 16px;margin-bottom:16px;border-radius:0 6px 6px 0;}
.tt{font-size:13pt;font-weight:700;color:#1e3a5f;margin-bottom:2px;}
.tsub{font-size:8.5pt;color:#6b7280;margin-bottom:10px;}
.cr{display:flex;gap:14px;margin-bottom:10px;align-items:center;}
.cb{flex:1;text-align:center;}
.cb img{max-width:100%;height:auto;}
.cl{font-size:7.5pt;color:#6b7280;margin-top:3px;}
.alloc{text-align:center;margin:10px 0;}
.alloc img{max-width:55%;height:auto;}
.rsn{font-size:8pt;color:#374151;line-height:1.45;}
.ftr{margin-top:36px;padding-top:10px;border-top:0.5px solid #e2e8f0;font-size:7.5pt;color:#9ca3af;text-align:center;}
</style>
</head><body>

<div class="hdr">
  <div class="hdr-brand">AI HEDGE FUND</div>
  <div class="hdr-meta">
    <div class="fn">{{ flow_name }}</div>
    {% if start_date or end_date %}<div>{{ start_date }}{% if start_date and end_date %} — {% endif %}{{ end_date }}</div>{% endif %}
    <div>Generated {{ generated_at }}</div>
  </div>
</div>

<section>
<h1>Executive Summary</h1>
<table>
<thead><tr><th>Ticker</th><th>Price</th><th>Action</th><th>Qty</th><th>Confidence</th><th>PM Reasoning</th></tr></thead>
<tbody>
{% for row in summary_rows %}
<tr>
  <td><strong>{{ row.ticker }}</strong></td>
  <td>{{ row.price }}</td>
  <td><span class="badge b-{{ row.action }}">{{ row.action | upper }}</span></td>
  <td>{{ row.quantity }}</td>
  <td>{{ row.confidence }}</td>
  <td class="rsn">{{ row.reasoning | e }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</section>

{% if alloc_chart %}
<section>
<h1>Portfolio Allocation</h1>
<div class="alloc"><img src="data:image/png;base64,{{ alloc_chart }}" alt="Portfolio allocation chart"/></div>
</section>
{% endif %}

<section>
<h1>Ticker Analysis</h1>
{% for t in ticker_sections %}
<div class="ts">
  <div class="tt">{{ t.ticker }}</div>
  <div class="tsub">{{ t.price }} &nbsp;|&nbsp; {{ t.agent_count }} analyst signal{{ 's' if t.agent_count != 1 else '' }}</div>

  {% if t.sparkline or t.donut %}
  <div class="cr">
    {% if t.sparkline %}
    <div class="cb" style="flex:2;">
      <img src="data:image/png;base64,{{ t.sparkline }}" alt="{{ t.ticker }} price history"/>
      <div class="cl">90-day price history</div>
    </div>
    {% endif %}
    {% if t.donut %}
    <div class="cb" style="flex:1;">
      <img src="data:image/png;base64,{{ t.donut }}" alt="{{ t.ticker }} signal distribution"/>
      <div class="cl">Signal distribution</div>
    </div>
    {% endif %}
  </div>
  {% endif %}

  {% if t.agents %}
  <table>
  <thead><tr><th style="width:20%">Analyst</th><th style="width:9%">Signal</th><th style="width:9%">Conf.</th><th>Reasoning</th></tr></thead>
  <tbody>
  {% for a in t.agents %}
  <tr>
    <td><strong>{{ a.name }}</strong></td>
    <td><span class="badge b-{{ a.signal }}">{{ a.signal | upper }}</span></td>
    <td>{{ a.confidence }}%</td>
    <td class="rsn">{{ a.reasoning | e }}</td>
  </tr>
  {% endfor %}
  </tbody>
  </table>
  {% endif %}
</div>
{% endfor %}
</section>

<div class="ftr">AI Hedge Fund Platform &nbsp;·&nbsp; This report is for informational purposes only and does not constitute financial advice.</div>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf(run_data: Dict[str, Any]) -> bytes:
    """
    Generate a PDF report from pipeline run data.
    Synchronous — call via asyncio.run_in_executor.
    """
    decisions: Dict = run_data.get("decisions", {})
    analyst_signals: Dict = run_data.get("analyst_signals", {})
    current_prices: Dict = run_data.get("current_prices", {})
    tickers: List[str] = run_data.get("tickers") or list(decisions.keys())
    flow_name: str = run_data.get("flow_name") or "AI Hedge Fund Report"
    start_date: str = run_data.get("start_date") or ""
    end_date: str = run_data.get("end_date") or ""

    # Filter out risk manager signals
    filtered_signals = {
        k: v for k, v in analyst_signals.items()
        if not k.lower().startswith("risk_management")
    }

    # Build summary rows
    summary_rows = []
    for t in tickers:
        d = decisions.get(t, {})
        reasoning = str(d.get("reasoning", "") or "")
        short_rsn = (reasoning[:140] + "…") if len(reasoning) > 140 else reasoning
        price_val = float(current_prices.get(t, 0))
        summary_rows.append({
            "ticker": t,
            "price": f"${price_val:.2f}",
            "action": str(d.get("action", "hold")).lower(),
            "quantity": d.get("quantity", 0),
            "confidence": f"{float(d.get('confidence', 0)):.1f}%",
            "reasoning": short_rsn,
        })

    # Portfolio allocation chart
    alloc_chart = _chart_allocation_pie(decisions, current_prices)

    # Per-ticker sections
    ticker_sections = []
    for t in tickers:
        # Price sparkline — re-fetch from api
        sparkline = ""
        try:
            from src.tools.api import get_prices
            end = end_date or datetime.now().strftime("%Y-%m-%d")
            start = start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            prices = get_prices(t, start, end)
            sparkline = _chart_sparkline(t, prices)
        except Exception as e:
            print(f"[pdf_service] price fetch failed for {t}: {e}")

        # Signal donut
        donut = _chart_signal_donut(t, filtered_signals)

        # Agent signal rows
        agents = []
        for agent_id, sig_data in filtered_signals.items():
            if t not in sig_data:
                continue
            s = sig_data[t]
            agents.append({
                "name": _pretty_agent_name(agent_id),
                "signal": str(s.get("signal", "neutral")).lower(),
                "confidence": round(float(s.get("confidence", 0)), 1),
                "reasoning": _pretty_reasoning(s.get("reasoning")),
            })
        agents.sort(key=lambda x: x["confidence"], reverse=True)

        ticker_sections.append({
            "ticker": t,
            "price": f"${float(current_prices.get(t, 0)):.2f}",
            "agent_count": len(agents),
            "sparkline": sparkline,
            "donut": donut,
            "agents": agents,
        })

    # Render HTML via Jinja2
    env = Environment(loader=BaseLoader(), autoescape=False)
    template = env.from_string(_HTML_TEMPLATE)
    html = template.render(
        flow_name=flow_name,
        start_date=start_date,
        end_date=end_date,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        summary_rows=summary_rows,
        alloc_chart=alloc_chart,
        ticker_sections=ticker_sections,
    )

    return WeasyHTML(string=html).write_pdf()