"""Offline HTML layout and numerical formatting for saved physics results."""
from html import escape
from math import isfinite


def esc(value):
    return escape(str(value), quote=True)


def number(value):
    if value is None or value == '':
        return 'Not calculated'
    parsed = float(value)
    return format(parsed, '.7g') if isfinite(parsed) else 'Unavailable'


def measure(value, error):
    if value is None or value == '':
        return 'Not calculated'
    return number(value) + (' ± ' + number(error) if error not in (None, '') else '')


def percent(value):
    return number(100 * float(value)) + '%' if value not in (None, '') else 'Unavailable'


def table(headers, rows):
    head = ''.join('<th scope="col">' + esc(h) + '</th>' for h in headers)
    body = ''.join('<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def section(anchor, title, body):
    return '<section id="' + esc(anchor) + '"><h2>' + esc(title) + '</h2>' + body + '</section>'


STYLE = '''
:root { color-scheme:light; --ink:#182e39; --teal:#00685f; --line:#c6d6d4; --muted:#4a6067; }
* { box-sizing:border-box } html {scroll-behavior:smooth} body {margin:0;background:#f4f7f6;color:var(--ink);font:16px/1.55 system-ui,sans-serif} main {max-width:78rem;margin:2rem auto;padding:0 clamp(.85rem,4vw,3.5rem) 3rem;background:white;border-top:5px solid var(--teal)}
header {padding:2rem 0 1rem} h1,h2 {font-family:Georgia,serif;line-height:1.17} h1 {font-size:clamp(2.2rem,5vw,3.6rem);margin:.25rem 0 .65rem;overflow-wrap:anywhere} h2 {font-size:1.85rem;margin:0 0 .8rem;color:#104e4a} h3 {font-size:1.05rem;margin:1.3rem 0 .45rem} p {margin:.5rem 0 .85rem} .meta,.note {color:var(--muted)} nav {display:flex;gap:.6rem 1.2rem;flex-wrap:wrap;border-block:1px solid var(--line);padding:.8rem 0} a {color:var(--teal);text-underline-offset:.2em} a:focus-visible,summary:focus-visible,select:focus-visible {outline:3px solid #da851f;outline-offset:2px} .download {display:inline-block;background:#075e58;color:white;padding:.45rem .85rem;border-radius:.25rem;text-decoration:none;font-weight:650}
section {padding:1.5rem 0 .6rem} .scroll {max-width:100%;overflow-x:auto;margin:.6rem 0 1rem} table {width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.91rem;text-align:left} td,th {padding:.55rem .65rem;vertical-align:top} th {background:#203c49;color:white;font-weight:610} td {border-bottom:1px solid var(--line);overflow-wrap:anywhere} tr:nth-child(even) td {background:#f0f7f5} td strong {white-space:nowrap} summary {cursor:pointer;color:#145952} details {margin:.4rem 0} details pre,code {white-space:pre-wrap;overflow-wrap:anywhere} .attention {border-left:4px solid #ae6a1b;background:#fff6e8;padding:.6rem .9rem} select {padding:.35rem;margin:.25rem .6rem .4rem 0;max-width:100%} ul {padding-left:1.3rem} .filters {display:flex;flex-wrap:wrap;gap:.3rem 1rem} [hidden] {display:none!important}
@media(max-width:640px){main{margin:0 auto}header{padding-top:1rem}td,th{padding:.4rem}.scroll table{min-width:30rem}}
@media print {body,main{background:white;margin:0;padding:0}main{max-width:none;border:0}nav,.filters,.download{display:none}.scroll{overflow:visible}.scroll table{min-width:0;font-size:8pt}th{color:black;background:#e0e9e7}section{break-inside:auto}h2,h3{break-after:avoid}tr{break-inside:avoid}thead{display:table-header-group}a{color:inherit}}
'''

def document(title, body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title><style>' + STYLE +
            '</style></head><body><main>' + body + '</main></body></html>\n')
