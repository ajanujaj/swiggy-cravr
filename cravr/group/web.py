"""The poll page each group member opens: /group/<id>.

Plain server-rendered HTML (no JS needed) so it works in any phone
browser from a WhatsApp link. Mounted by server.py before Chainlit.
"""

from __future__ import annotations

import html

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .combine import CUISINE_CHOICES
from .store import add_response, get_group

router = APIRouter()

BUDGETS = [(150, "Up to ₹150"), (250, "Up to ₹250"), (400, "Up to ₹400"), (600, "Up to ₹600"), (1000, "Up to ₹1000")]

_STYLE = """
:root{--bg:#fff7f2;--card:#fff;--ink:#1f1a17;--muted:#6b625c;--accent:#ff5200;--line:#f0e2d8}
@media (prefers-color-scheme:dark){:root{--bg:#171311;--card:#221c19;--ink:#f5efe9;--muted:#b3a79e;--line:#3a302a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:520px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:24px;margin:8px 0 4px}p.sub{color:var(--muted);margin:0 0 18px}
fieldset{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:14px 14px 10px;margin:0 0 14px}
legend{font-weight:700;padding:0 6px}
label.opt{display:inline-flex;align-items:center;gap:6px;margin:0 8px 8px 0;padding:7px 12px;border:1px solid var(--line);border-radius:999px;cursor:pointer}
label.opt input{accent-color:var(--accent)}
input[type=text]{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:transparent;color:var(--ink);font:inherit}
button{width:100%;padding:13px;border:0;border-radius:12px;background:var(--accent);color:#fff;font:700 16px system-ui;cursor:pointer}
.done{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;text-align:center}
"""


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>",
        status_code=status,
    )


def _choices(name: str, options: list[tuple[str, str]], kind: str = "radio", checked: str | None = None) -> str:
    return "".join(
        f"<label class='opt'><input type='{kind}' name='{name}' value='{html.escape(value)}'"
        f"{' checked' if value == checked else ''}> {html.escape(label)}</label>"
        for value, label in options
    )


@router.get("/group/{group_id}", response_class=HTMLResponse)
async def group_form(group_id: str) -> HTMLResponse:
    group = get_group(group_id)
    if not group:
        return _page("Poll not found", "<h1>This poll has expired</h1><p class='sub'>Ask for a new link.</p>", 404)
    answered = len(group.get("responses") or [])
    cuisines = [(c, c) for c in CUISINE_CHOICES]
    body = f"""
<h1>What should we order?</h1>
<p class='sub'>Group order for {group['size']} · {answered} answered so far. Takes 30 seconds.</p>
<form method='post'>
  <fieldset><legend>Your name</legend><input type='text' name='name' maxlength='40' required placeholder='e.g. Riya'></fieldset>
  <fieldset><legend>You eat</legend>{_choices('diet', [('veg', 'Veg only'), ('nonveg', 'Non-veg'), ('either', 'Either')], checked='either')}</fieldset>
  <fieldset><legend>Your budget</legend>{_choices('budget', [(str(v), l) for v, l in BUDGETS], checked='400')}</fieldset>
  <fieldset><legend>Spice</legend>{_choices('spice', [('mild', 'Mild'), ('medium', 'Medium'), ('spicy', 'Spicy')], checked='medium')}</fieldset>
  <fieldset><legend>In the mood for</legend>{_choices('likes', cuisines, 'checkbox')}</fieldset>
  <fieldset><legend>Not today, please</legend>{_choices('dislikes', cuisines, 'checkbox')}</fieldset>
  <fieldset><legend>Allergies (optional)</legend><input type='text' name='allergies' maxlength='80' placeholder='e.g. peanuts'></fieldset>
  <button type='submit'>Send my answer</button>
</form>"""
    return _page("Group order poll", body)


@router.post("/group/{group_id}", response_class=HTMLResponse)
async def group_submit(group_id: str, request: Request) -> HTMLResponse:
    form = await request.form()
    allowed = set(CUISINE_CHOICES)
    budget = str(form.get("budget") or "")
    response = {
        "name": str(form.get("name") or "").strip()[:40] or "Someone",
        "diet": str(form.get("diet") or "either") if form.get("diet") in ("veg", "nonveg", "either") else "either",
        "budget": budget if budget.isdigit() else "",
        "spice": str(form.get("spice") or "medium") if form.get("spice") in ("mild", "medium", "spicy") else "medium",
        "likes": [c for c in form.getlist("likes") if c in allowed],
        "dislikes": [c for c in form.getlist("dislikes") if c in allowed],
        "allergies": str(form.get("allergies") or "").strip()[:80],
    }
    group = add_response(group_id, response)
    if not group:
        return _page("Poll not found", "<h1>This poll has expired</h1>", 404)
    answered = len(group.get("responses") or [])
    return _page(
        "Thanks!",
        f"<div class='done'><h1>Thanks, {html.escape(response['name'])}!</h1>"
        f"<p class='sub'>{answered} of {group['size']} have answered. The organiser will pick a place that works for everyone.</p></div>",
    )
