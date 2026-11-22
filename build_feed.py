#!/usr/bin/env python3
"""
Builds a longevity & wellness article feed from publishers you pick in config.yaml.

Outputs (in ./docs):
  index.html     - browsable feed page (filter by publisher, topic, keyword)
  feed.xml       - RSS feed you can subscribe to in any reader
  articles.json  - rolling archive (so articles + summaries persist between runs)

Run:  python build_feed.py
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import feedparser
import requests
import yaml

ROOT = Path(__file__).parent
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)
ARCHIVE = OUT / "articles.json"
UA = {"User-Agent": "longevity-feed/1.0 (personal research feed)"}


# ---------------------------------------------------------------- helpers
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def entry_date(e):
    for key in ("published_parsed", "updated_parsed"):
        t = e.get(key)
        if t:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
    return datetime.now(timezone.utc)


def term_regex(terms):
    parts = [re.escape(t) for t in terms]
    return re.compile(r"(?<!\w)(" + "|".join(parts) + r")(?!\w)", re.I)


def make_tagger(cfg):
    kw_re = term_regex(cfg["keywords"])
    topic_res = {name: term_regex(terms) for name, terms in (cfg.get("topics") or {}).items()}

    def tag(text):
        kws = sorted({m.lower() for m in kw_re.findall(text)})
        topics = [name for name, rx in topic_res.items() if rx.search(text)]
        return kws, topics

    return tag


# ---------------------------------------------------------------- summaries
FINDING_WORDS = re.compile(
    r"\b(we (found|show|demonstrate|report|identif|reveal|observ)|results? (show|indicate|suggest|reveal)|"
    r"(found|shows?|demonstrates?|reveals?|suggests?|indicates?|conclude[sd]?|associated with|"
    r"improved|reduced|increased|extended|protects?))\b",
    re.I,
)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def is_junk_abstract(text):
    """Some RSS feeds only give 'Journal, Published online: ...; doi:...' instead of an abstract."""
    return (not text) or (len(text) < 250 and re.search(r"Published online|doi:\s*10\.|^\s*Volume \d+", text, re.I))


def clip(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def summarize_extractive(text, n=2, max_chars=320):
    if is_junk_abstract(text):
        return ""
    sents = split_sentences(text)
    if len(sents) <= n:
        return clip(" ".join(sents), max_chars)
    scored = [i for i, s in enumerate(sents) if FINDING_WORDS.search(s)]
    # prefer finding-like sentences from the back half (results/conclusions), else first n
    picks = (scored[-n:] if len(scored) >= 1 else list(range(n)))[:n]
    picks = sorted(set(picks)) or list(range(n))
    return clip(" ".join(sents[i] for i in picks), max_chars)


def summarize_with_claude(articles, scfg):
    """Summarize only articles that don't have a Claude summary yet. Falls back silently."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  (summary.mode=claude but ANTHROPIC_API_KEY not set; using extractive)", file=sys.stderr)
        return
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as ex:
        print(f"  ! anthropic SDK unavailable ({ex}); using extractive", file=sys.stderr)
        return
    todo = [a for a in articles
            if a.get("summary_source") != "claude" and not is_junk_abstract(a.get("abstract", ""))]
    todo = todo[: scfg.get("max_new_per_run", 60)]
    done = 0
    for a in todo:
        prompt = (
            "Summarize this journal article in 1-2 plain-language sentences for a biology student. "
            "State the main finding or what was done. Use ONLY the title and abstract below; "
            "do not add facts that aren't there. Output just the summary.\n\n"
            f"Title: {a['title']}\n\nAbstract: {a['abstract']}"
        )
        try:
            msg = client.messages.create(
                model=scfg.get("model", "claude-haiku-5-5"),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "text", None)).strip()
            if text:
                a["summary_short"] = text
                a["summary_source"] = "claude"
                done += 1
        except Exception as ex:
            print(f"  ! summary failed for '{a['title'][:40]}…': {ex}", file=sys.stderr)
    print(f"  Claude summaries written: {done}")


def ensure_summaries(articles, cfg):
    scfg = cfg.get("summary", {})
    n, mc = scfg.get("sentences", 2), scfg.get("max_chars", 320)
    for a in articles:
        if not a.get("summary_short"):
            a["summary_short"] = summarize_extractive(a.get("abstract") or a.get("summary", ""), n, mc)
            a["summary_source"] = "extractive"
    if scfg.get("mode") == "claude":
        summarize_with_claude(articles, scfg)


# ---------------------------------------------------------------- fetchers
def fetch_rss(publisher, feed, tag, require_keyword):
    items = []
    try:
        resp = requests.get(feed["url"], headers=UA, timeout=30)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as ex:
        print(f"  ! {feed['name']}: {ex}", file=sys.stderr)
        return items
    if not parsed.entries:
        print(f"  ! {feed['name']}: no entries (URL may have changed)", file=sys.stderr)
    for e in parsed.entries:
        title = strip_html(e.get("title", ""))
        abstract = strip_html(e.get("summary", e.get("description", "")))
        kws, topics = tag(f"{title} {abstract}")
        if require_keyword and not feed.get("on_topic") and not kws and not topics:
            continue
        link = e.get("link", "")
        if not link:
            continue
        items.append({
            "id": e.get("id") or link,
            "title": title,
            "link": link,
            "abstract": abstract[:1500],
            "journal": feed["name"],
            "publisher": publisher,
            "date": entry_date(e).isoformat(),
            "tags": kws[:5],
            "topics": topics,
        })
    print(f"  {feed['name']}: {len(items)} kept")
    return items


def fetch_europepmc(publisher, q, tag, max_age_days):
    since = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    query = f"({q['query']}) AND (FIRST_PDATE:[{since} TO 3000-12-31])"
    items = []
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": 100,
                    "resultType": "core", "sort": "P_PDATE_D desc"},
            headers=UA, timeout=30,
        )
        r.raise_for_status()
        for a in r.json().get("resultList", {}).get("result", []):
            doi = a.get("doi")
            link = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{a.get('source')}/{a.get('id')}"
            date = a.get("firstPublicationDate") or a.get("firstIndexDate") or ""
            title = strip_html(a.get("title", ""))
            abstract = strip_html(a.get("abstractText", ""))
            kws, topics = tag(f"{title} {abstract}")
            items.append({
                "id": doi or link,
                "title": title,
                "link": link,
                "abstract": abstract[:1500],
                "journal": a.get("journalTitle") or q["name"],
                "publisher": publisher,
                "date": (date + "T00:00:00+00:00") if len(date) == 10 else datetime.now(timezone.utc).isoformat(),
                "tags": kws[:5],
                "topics": topics,
            })
    except Exception as ex:
        print(f"  ! {q['name']}: {ex}", file=sys.stderr)
    print(f"  {q['name']}: {len(items)} kept")
    return items


def topic_queries(cfg):
    """One Europe PMC query per tracked topic, built from its terms."""
    out = []
    for name, terms in (cfg.get("topics") or {}).items():
        ors = " OR ".join(f'"{t}"' for t in terms)
        out.append({"name": f"{name} (Europe PMC)",
                    "query": f'({ors}) AND (PUB_TYPE:"Journal Article") AND SRC:MED'})
    return out


# ---------------------------------------------------------------- archive + output
def load_archive():
    if ARCHIVE.exists():
        try:
            return {a["id"]: a for a in json.loads(ARCHIVE.read_text())}
        except Exception:
            pass
    return {}


def write_html(articles, topic_names):
    publishers = sorted({a["publisher"] for a in articles})
    cards = []
    for a in articles:
        d = a["date"][:10]
        topic_chips = "".join(f'<span class="topic">{escape(t)}</span>' for t in a.get("topics", []))
        tags = "".join(f'<span class="tag">{escape(t)}</span>' for t in a["tags"])
        summ = a.get("summary_short", "")
        summ_html = f'<p class="sum">{escape(summ)}</p>' if summ else '<p class="sum none">No abstract in feed. Open the article for details.</p>'
        cards.append(f"""
<article data-pub="{escape(a['publisher'])}" data-topics="{escape('|'.join(a.get('topics', [])))}">
  <div class="meta">{escape(a['journal'])} &middot; {d}</div>
  <h2><a href="{escape(a['link'])}" target="_blank" rel="noopener">{escape(a['title'])}</a></h2>
  {summ_html}
  <div>{topic_chips}{tags}</div>
</article>""")
    pub_opts = "".join(f"<option>{escape(p)}</option>" for p in publishers)
    topic_opts = "".join(f"<option>{escape(t)}</option>" for t in topic_names)
    updated = datetime.now().strftime("%b %d, %Y %H:%M")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Longevity &amp; Wellness Feed</title>
<link rel="alternate" type="application/rss+xml" href="feed.xml">
<style>
  :root {{ --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e6e4dc; --accent:#2f6f5e; --chip:#e4f1ec; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecebe6; --muted:#9a9990; --card:#1f1f1c; --line:#2f2f2a; --accent:#6fc2aa; --chip:#23352f; }} }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,sans-serif; }}
  main {{ max-width:760px; margin:0 auto; padding:24px 16px 64px; }}
  h1 {{ margin:0 0 4px; }} .sub {{ color:var(--muted); margin-bottom:20px; }}
  .controls {{ display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }}
  input,select {{ padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--fg); font:inherit; }}
  input {{ flex:1; min-width:180px; }}
  article {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin-bottom:12px; }}
  article h2 {{ font-size:1.05rem; margin:4px 0 6px; }}
  article a {{ color:var(--fg); text-decoration:none; }} article a:hover {{ color:var(--accent); text-decoration:underline; }}
  .sum {{ margin:0 0 8px; font-size:.93rem; }} .sum.none {{ color:var(--muted); font-style:italic; }}
  .meta {{ font-size:.8rem; color:var(--accent); font-weight:600; }}
  .tag {{ font-size:.72rem; border:1px solid var(--line); border-radius:99px; padding:1px 8px; margin-right:4px; color:var(--muted); }}
  .topic {{ font-size:.72rem; background:var(--chip); color:var(--accent); font-weight:600; border-radius:99px; padding:2px 9px; margin-right:4px; }}
</style></head><body><main>
<h1>Longevity &amp; Wellness Feed</h1>
<div class="sub">{len(articles)} articles &middot; updated {updated} &middot; <a href="feed.xml" style="color:var(--accent)">RSS</a></div>
<div class="controls">
  <input id="q" placeholder="Filter by keyword…">
  <select id="topic"><option value="">All topics</option>{topic_opts}</select>
  <select id="pub"><option value="">All publishers</option>{pub_opts}</select>
</div>
<div id="list">{''.join(cards)}</div>
<script>
  const q=document.getElementById('q'), pub=document.getElementById('pub'), topic=document.getElementById('topic');
  const items=[...document.querySelectorAll('article')];
  function apply(){{
    const t=q.value.toLowerCase(), p=pub.value, tp=topic.value;
    items.forEach(a=>{{
      const ok = a.textContent.toLowerCase().includes(t)
        && (!p || a.dataset.pub===p)
        && (!tp || a.dataset.topics.split('|').includes(tp));
      a.style.display = ok ? '' : 'none';
    }});
  }}
  q.oninput=apply; pub.onchange=apply; topic.onchange=apply;
</script></main></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_rss(articles):
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    items = []
    for a in articles[:100]:
        dt = datetime.fromisoformat(a["date"]).strftime("%a, %d %b %Y %H:%M:%S +0000")
        cats = "".join(f"<category>{xml_escape(c)}</category>" for c in [a["journal"], *a.get("topics", [])])
        items.append(f"""<item>
<title>{xml_escape(a['title'])}</title>
<link>{xml_escape(a['link'])}</link>
<guid isPermaLink="false">{xml_escape(a['id'])}</guid>
<pubDate>{dt}</pubDate>
{cats}
<description>{xml_escape(a.get('summary_short', ''))}</description>
</item>""")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Longevity &amp; Wellness Feed</title>
<link>index.html</link>
<description>Curated journal articles on longevity and wellness</description>
<lastBuildDate>{now}</lastBuildDate>
{''.join(items)}
</channel></rss>"""
    (OUT / "feed.xml").write_text(rss, encoding="utf-8")


def main():
    cfg = load_config()
    tag = make_tagger(cfg)
    require_kw = cfg.get("require_keyword", True)
    max_age = cfg.get("max_age_days", 30)

    archive = load_archive()
    new_items = []
    for pub in cfg["publishers"]:
        if not pub.get("enabled", True):
            continue
        print(pub["name"])
        for feed in pub.get("feeds", []):
            new_items += fetch_rss(pub["name"], feed, tag, require_kw)
        if pub.get("europepmc_queries") is not None:
            for q in [*pub["europepmc_queries"], *topic_queries(cfg)]:
                new_items += fetch_europepmc(pub["name"], q, tag, max_age)

    for it in new_items:
        archive.setdefault(it["id"], it)  # dedupe by id/DOI

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age)
    articles = [a for a in archive.values() if datetime.fromisoformat(a["date"]) >= cutoff]
    articles.sort(key=lambda a: a["date"], reverse=True)
    articles = articles[: cfg.get("max_items", 300)]

    ensure_summaries(articles, cfg)

    ARCHIVE.write_text(json.dumps(articles, indent=1))
    write_html(articles, list((cfg.get("topics") or {}).keys()))
    write_rss(articles)
    print(f"\nDone: {len(articles)} articles -> {OUT}/index.html")


if __name__ == "__main__":
    main()
