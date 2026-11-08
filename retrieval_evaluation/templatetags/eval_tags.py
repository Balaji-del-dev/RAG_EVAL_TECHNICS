"""Formatting helpers for metric values in templates."""

from django import template

from ..analysis.failure_analysis import ISSUES, STATUS_LABELS

register = template.Library()

DASH = "—"


@register.filter
def metric_value(value, display="percent"):
    """0.4 -> '40.0%' (percent) or '0.400' (decimal). 'percent-compact' drops a trailing .0 (40%)."""
    if value is None or value == "":
        return DASH
    value = float(value)
    if display.startswith("percent"):
        text = f"{value * 100:.1f}"
        if display == "percent-compact" and text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"
    return f"{value:.3f}"


@register.filter
def signed_delta(value, display="percent"):
    """Signed change: '+40.0 pts' for percent metrics, '+0.400' for decimal ones."""
    if value is None:
        return DASH
    value = float(value)
    if abs(value) < 5e-5:
        return "0.0 pts" if display.startswith("percent") else "0.000"
    sign = "+" if value > 0 else "−"
    if display.startswith("percent"):
        return f"{sign}{abs(value) * 100:.1f} pts"
    return f"{sign}{abs(value):.3f}"


@register.filter
def bar_width(value):
    """A 0-1 score as a CSS percentage, clamped (not localised, safe for style attributes)."""
    if value is None:
        return "0"
    return f"{max(0.0, min(1.0, float(value))) * 100:.2f}"


@register.filter
def get_item(mapping, key):
    if mapping is None:
        return None
    return mapping.get(key)


@register.filter
def metric_label(metric, k):
    return metric.label(k)


@register.filter
def issue_label(code):
    issue = ISSUES.get(code)
    return issue.label if issue else code


@register.filter
def issue_metric(code):
    issue = ISSUES.get(code)
    return issue.metric if issue else ""


@register.filter
def status_label(code):
    return STATUS_LABELS.get(code, code)


@register.filter
def ms(value):
    if value is None:
        return DASH
    value = float(value)
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


@register.filter
def grade_label(grade):
    return {3: "Highly relevant", 2: "Relevant", 1: "Somewhat relevant"}.get(grade, "Not relevant")


@register.simple_tag
def query_string(request, **changes):
    """The current query string with some parameters changed or removed (value '')."""
    params = request.GET.copy()
    for key, value in changes.items():
        if value in ("", None):
            params.pop(key, None)
        else:
            params[key] = value
    encoded = params.urlencode()
    return f"?{encoded}" if encoded else "?"
