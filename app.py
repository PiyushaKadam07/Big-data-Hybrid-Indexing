"""
Hybrid Indexing System - Streamlit app
CS63016 · Kent State University
Piyusha Kadam

Run with:  streamlit run app.py
"""

import html
import os
import random
import re
import statistics
import sys

import json

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from dataset_generator import TOPICS, generate_dataset  # noqa: E402
from hybrid_index import (  # noqa: E402
    SAMPLE_DOCS, TOKEN_PATTERN, Document, HybridIndexingSystem, sample_documents, tokenize, years_to_months,
)
from load_arxiv import submission_month  # noqa: E402

st.set_page_config(page_title="Hybrid Indexing", page_icon="🔎", layout="wide")

# ── Look and feel ─────────────────────────────────────────────────────────────

MODES = {
    "hybrid": {"name": "Hybrid", "color": "#0d9488",
               "how": "Checks the directory, then asks only the servers that can have a match."},
    "distributed": {"name": "Distributed", "color": "#ea580c",
                    "how": "Asks every server, every time."},
    "centralized": {"name": "Centralized", "color": "#7c3aed",
                    "how": "One big server holds everything."},
}
GREY, INK, MUTED = "#d1d5db", "#111827", "#6b7280"

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .stMarkdown p.headline { font-size: 1.35rem; line-height: 1.5; margin: 0.5rem 0 1rem; }
  .stMarkdown p.mode-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0; }
  .stMarkdown p.mode-how { color: #6b7280; font-size: 0.9rem; margin: 0 0 0.25rem; min-height: 3.4em; line-height: 1.6; }
  .stMarkdown p.mode-count { font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0 0; }
  .card { display: grid; grid-template-columns: 2.2rem 1fr; gap: 0.75rem; padding: 0.9rem 1rem;
          border: 1px solid #e5e7eb; border-radius: 12px; margin-bottom: 0.6rem; background: #fff; }
  .rank { font-weight: 700; font-size: 1.2rem; color: #0d9488; text-align: center; }
  .stMarkdown p.doc { font-size: 1.05rem; margin: 0 0 0.35rem; color: #111827; }
  .stMarkdown p.why { color: #4b5563; font-size: 0.9rem; margin: 0 0 0.4rem; }
  .meter { height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; max-width: 260px; }
  .meter span { display: block; height: 100%; background: #0d9488; }
  .chip { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #f3f4f6;
          color: #374151; font-size: 0.8rem; margin-left: 6px; }
  mark { background: #fde68a; color: inherit; padding: 0 2px; border-radius: 3px; }
  .stMarkdown p.big-stat { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0; }
  .stMarkdown p.big-label { color: #4b5563; margin: 0.2rem 0 0; }
</style>
""", unsafe_allow_html=True)


def server_name(node_id: str) -> str:
    """node_0 -> Server 1 (people count from one)."""
    return f"Server {int(node_id.split('_')[-1]) + 1}"


# ── Building the system ───────────────────────────────────────────────────────

PLACEMENTS = {
    "By topic (similar documents together)": "topic",
    "By date (each server holds a range of years)": "range",
    "Randomly (spread evenly by hashing)": "hash",
}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def month_name(month: int) -> str:
    """202104 -> "April 2021"."""
    return f"{MONTH_NAMES[month % 100 - 1]} {month // 100}"


def year_span(node_id: str) -> str:
    span = system.node_date_range(node_id)
    if not span:
        return "no dated documents"
    first, last = span[0] // 100, span[1] // 100
    return str(first) if first == last else f"{first}–{last}"


ARXIV_PATH = os.path.join(os.path.dirname(__file__), "data", "arxiv_sample.jsonl")
CATEGORY_NAMES = {
    "cs.AI": "Artificial intelligence", "cs.LG": "Machine learning", "cs.CL": "Language and NLP",
    "cs.CV": "Computer vision", "cs.CR": "Security", "cs.RO": "Robotics", "cs.DC": "Distributed computing",
    "cs.DB": "Databases", "cs.IR": "Information retrieval", "cs.SE": "Software engineering",
    "cs.NI": "Networking", "cs.HC": "Human-computer interaction", "cs.DS": "Algorithms",
    "cs.IT": "Information theory", "cs.SY": "Systems and control", "cs.NE": "Neural computing",
}


def topic_label(category: str) -> str:
    return CATEGORY_NAMES.get(category, category)


@st.cache_resource(show_spinner=False)
def load_arxiv_papers() -> list[dict]:
    with open(ARXIV_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Keep only the current index: each one for 20,000 papers takes about 200 MB,
# and free hosting has about 1 GB.
@st.cache_resource(show_spinner=False, max_entries=1)
def build_system(num_nodes: int, strategy: str, source: str) -> HybridIndexingSystem:
    if source == "arxiv":
        docs = [Document(
            doc_id=p["id"], content=f"{p['title']}. {p['abstract']}",
            metadata={"category": p["category"], "title": p["title"], "abstract": p["abstract"],
                      "month": p.get("month") or submission_month(p["id"]) or 0,
                      "authors": p.get("authors", ""), "url": f"https://arxiv.org/abs/{p['id']}"})
            for p in load_arxiv_papers()]
    else:
        docs = sample_documents()
    return HybridIndexingSystem.build(docs, num_nodes=num_nodes, strategy=strategy)


with st.sidebar:
    st.markdown("## 🔎 Hybrid Indexing")
    st.caption("CS63016 · Kent State University")
    st.markdown("---")
    page = st.radio("Go to", ["Try a search", "How it works", "What happens at scale", "Look inside the servers"])
    st.markdown("---")
    st.markdown("**Setup**")
    num_nodes = st.select_slider("Number of servers", options=[2, 4, 8, 16], value=4)
    placement_label = st.radio("How documents are placed on servers", list(PLACEMENTS), index=0,
                               help="Topic placement keeps related documents together, so word searches "
                                    "can skip servers. Date placement gives each server a range of years, "
                                    "so searches limited to certain years can skip servers.")
    strategy = PLACEMENTS[placement_label]
    if os.path.exists(ARXIV_PATH):
        source_label = st.radio("Documents to search",
                                [f"arXiv papers ({len(load_arxiv_papers()):,})", f"{len(SAMPLE_DOCS)} sample sentences"])
        source = "arxiv" if source_label.startswith("arXiv") else "sample"
    else:
        source = "sample"
        st.caption(f"Searching {len(SAMPLE_DOCS)} sample sentences. Run load_arxiv.py to search real papers.")
    st.markdown("---")
    st.markdown("Built by **Piyusha Kadam**")

with st.spinner("Building the index (takes a few seconds after changing the setup)…"):
    system = build_system(num_nodes, strategy, source)


# ── Drawing the journey of a search ───────────────────────────────────────────

def matches_per_node(terms: list[str], months=None, match_all: bool = True) -> dict[str, int]:
    """How many documents on each server match, using the same rules as the search."""
    def count(node):
        doc_sets = [set(node.inverted_index.get(t, {})) for t in dict.fromkeys(terms)]
        docs = set.intersection(*doc_sets) if match_all else set.union(*doc_sets)
        return sum(1 for d in docs if not months or months[0] <= node.doc_month.get(d, 0) <= months[1])
    return {nid: count(node) for nid, node in system.nodes.items()}


def journey_svg(mode: str, asked: set[str], hits: dict[str, int], total_hits: int,
                not_needed: set[str] = frozenset()) -> str:
    """A small picture: the search, (the directory), and which servers were asked."""
    color = MODES[mode]["color"]
    W, box_w, box_h, gap, per_row = 352, 76, 56, 8, 4
    parts, cx = [], W / 2
    top_y = 34  # room for the directory box (kept for every mode so the pictures line up)

    def text(x, y, words, size=12, weight=400, fill=INK):
        return (f"<text x='{x}' y='{y}' text-anchor='middle' font-size='{size}' "
                f"font-weight='{weight}' fill='{fill}'>{html.escape(words)}</text>")

    parts.append(f"<rect x='{cx - 60}' y='4' width='120' height='26' rx='13' fill='{INK}'/>")
    parts.append(text(cx, 22, "Your search", 14, 600, "#ffffff"))
    source_y = 30
    if mode == "hybrid":
        parts.append(f"<line x1='{cx}' y1='30' x2='{cx}' y2='44' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<rect x='{cx - 70}' y='44' width='140' height='26' rx='6' fill='#ffffff' "
                     f"stroke='{color}' stroke-width='1.5'/>")
        parts.append(text(cx, 62, "Directory", 14, 600, color))
        source_y = 70

    if mode == "centralized":
        y = 70 + top_y
        parts.append(f"<line x1='{cx}' y1='{source_y}' x2='{cx}' y2='{y}' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<rect x='{cx - 120}' y='{y}' width='240' height='{box_h}' rx='8' fill='{color}' "
                     f"fill-opacity='0.12' stroke='{color}' stroke-width='1.5'/>")
        parts.append(text(cx, y + 23, "One big server", 14, 600))
        parts.append(text(cx, y + 42, f"all {system.total_docs:,} documents · {total_hits:,} found", 13, 400, MUTED))
        height = 92 + top_y + box_h + 8
    else:
        ids = list(system.nodes)
        rows = [ids[i:i + per_row] for i in range(0, len(ids), per_row)]
        y0 = 92 + top_y
        one_row = len(rows) == 1
        if not one_row:
            # Several rows: one arrow into a panel holding all the servers, so lines never cross boxes.
            panel_top, panel_bottom = y0 - 14, y0 + len(rows) * (box_h + 26) - 26 + 14
            parts.append(f"<line x1='{cx}' y1='{source_y}' x2='{cx}' y2='{panel_top}' stroke='{color}' stroke-width='2'/>")
            parts.append(f"<rect x='{(W - per_row * box_w - (per_row - 1) * gap) / 2 - 8}' y='{panel_top}' "
                         f"width='{per_row * box_w + (per_row - 1) * gap + 16}' height='{panel_bottom - panel_top}' "
                         f"rx='10' fill='#ffffff' stroke='{color}' stroke-width='1.5'/>")
        for r, row in enumerate(rows):
            row_w = len(row) * box_w + (len(row) - 1) * gap
            for c, nid in enumerate(row):
                x = (W - row_w) / 2 + c * (box_w + gap)
                y = y0 + r * (box_h + 26)
                if nid in asked:
                    if one_row:
                        parts.append(f"<line x1='{cx}' y1='{source_y}' x2='{x + box_w / 2}' y2='{y}' "
                                     f"stroke='{color}' stroke-width='1.5' opacity='0.8'/>")
                    parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='8' fill='{color}' "
                                 f"fill-opacity='0.12' stroke='{color}' stroke-width='1.5'/>")
                    parts.append(text(x + box_w / 2, y + 23, server_name(nid), 13, 600))
                    found = hits.get(nid, 0)
                    parts.append(text(x + box_w / 2, y + 42, f"{found:,} found", 13, 400, MUTED))
                else:
                    parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='8' fill='#f9fafb' "
                                 f"stroke='{GREY}' stroke-dasharray='4 3'/>")
                    parts.append(text(x + box_w / 2, y + 23, server_name(nid), 13, 600, "#9ca3af"))
                    label = "not needed" if nid in not_needed else "skipped"
                    parts.append(text(x + box_w / 2, y + 42, label, 13, 400, "#9ca3af"))
        height = y0 + len(rows) * (box_h + 26) - 18 + (0 if one_row else 14)
    return (f"<svg viewBox='0 0 {W} {height}' width='100%' role='img' "
            f"aria-label='{MODES[mode]['name']} search path' style='max-width:{W}px'>{''.join(parts)}</svg>")


def highlight(content: str, terms: set[str]) -> str:
    """Wrap the words that matched the search in <mark>."""
    out, last, lower = [], 0, content.lower()
    for m in TOKEN_PATTERN.finditer(lower):
        if m.group() in terms:
            out.append(html.escape(content[last:m.start()]))
            out.append(f"<mark>{html.escape(content[m.start():m.end()])}</mark>")
            last = m.end()
    out.append(html.escape(content[last:]))
    return "".join(out)


def snippet(text: str, terms: set[str], width: int = 280) -> str:
    """A short piece of a long abstract around the first matching word, with matches highlighted."""
    lower = text.lower()
    first = next((m.start() for m in TOKEN_PATTERN.finditer(lower) if m.group() in terms), 0)
    start = max(0, first - width // 3)
    if start:
        start = text.find(" ", start) + 1
    end = min(len(text), start + width)
    if end < len(text):
        end = text.rfind(" ", start, end)
    piece = highlight(text[start:end], terms)
    return ("…" if start else "") + piece + ("…" if end < len(text) else "")


def explain(doc: Document, terms: list[str]) -> str:
    """Why a document is in the results, in plain words."""
    stats = system.metadata.stats()
    present = [t for t in dict.fromkeys(terms) if t in set(tokenize(doc.content))]
    words = " and ".join(f"<b>{html.escape(t)}</b>" for t in present)
    why = f"Contains {words}"
    rarest = min(present, key=lambda t: stats.doc_freq.get(t, 0)) if present else None
    if rarest and len(set(terms)) > 1:
        n = stats.doc_freq.get(rarest, 0)
        why += f" · “{html.escape(rarest)}” is a rare word here ({n:,} of {stats.doc_count:,} documents), so it counts more"
    return why


# ── Page: Try a search ────────────────────────────────────────────────────────

def page_search():
    st.title("Find documents spread across many servers")
    st.markdown(
        "Large collections are too big for one computer, so they're split across several **servers**. "
        "This compares three ways of searching them. All three find the **same documents** — "
        "the difference is how much work it takes.")

    if source == "arxiv":
        examples = ["graph neural networks", "differential privacy", "robot grasping",
                    "query optimization", "speech recognition", "federated learning"]
    else:
        examples = ["consistent hashing", "NoSQL databases", "fault tolerance replication",
                    "distributed indexing nodes", "real time indexing", "B-Tree indexes"]

    def pick():
        if st.session_state.example:
            st.session_state.query = st.session_state.example

    if st.session_state.get("query_source") != source:
        st.session_state.query = examples[0]
        st.session_state.query_source = source
    st.text_input("What are you looking for?", key="query")
    st.pills("Or try one of these", examples, key="example", on_change=pick)

    match_label = st.radio("Find documents containing", ["All of my words", "Any of my words"],
                           horizontal=True,
                           help="“All” works like Google: every word must appear. “Any” matches far more "
                                "documents, so the directory can rarely skip a server.")
    match_all = match_label == "All of my words"

    spans = [system.node_date_range(n) for n in system.nodes]
    spans = [sp for sp in spans if sp]
    months, when = None, ""
    if spans:
        first_year = min(sp[0] for sp in spans) // 100
        last_year = max(sp[1] for sp in spans) // 100
        if st.toggle("Only show documents from certain years") and first_year < last_year:
            years = st.slider("Years", first_year, last_year, (max(first_year, last_year - 2), last_year))
            months = years_to_months(*years)
            when = f" from {years[0]}" if years[0] == years[1] else f" from {years[0]}–{years[1]}"

    prune = st.toggle("Skip servers that can't change the top 5", value=True,
                      help="The directory also knows each server's best possible score. Hybrid asks the most "
                           "promising server first, then skips servers that couldn't beat its top 5. "
                           "The top 5 results stay exactly the same.")

    query = st.session_state.query.strip()
    terms = tokenize(query)
    if not terms:
        st.info("Type a few words to search.")
        return

    runs = {"hybrid": system.hybrid_search(query, top_k=5, months=months, match_all=match_all, prune=prune),
            "distributed": system.distributed_search(query, top_k=5, months=months, match_all=match_all),
            "centralized": system.centralized_search(query, top_k=5, months=months, match_all=match_all)}
    found = runs["distributed"]["total_matches"]  # hybrid may stop early, so take the full count from here
    asked = runs["hybrid"]["nodes_queried"]
    by_score = len(runs["hybrid"]["skipped_by_score"])
    n = len(system.nodes)

    if found == 0:
        words = "all of those words" if match_all and len(set(terms)) > 1 else "those words"
        if asked == 0:
            headline = (f"No document{when} contains {words}. The directory knew that, "
                        f"so hybrid **didn't ask any server**.")
        else:
            headline = (f"No document{when} contains {words}. Hybrid asked **{asked} of {n} servers** "
                        f"that might have had one; the directory ruled out the rest.")
    elif asked < n:
        no_match = n - asked - by_score
        reasons = []
        if no_match:
            reasons.append(f"**{no_match}** {'have' if no_match != 1 else 'has'} no matching documents")
        if by_score:
            reasons.append(f"**{by_score}** couldn't beat the top 5")
        headline = (f"All three return the same top results from **{found:,} matching document"
                    f"{'s' if found != 1 else ''}{when}**. Hybrid asked **{asked} of {n} servers**: "
                    f"{' and '.join(reasons)}.")
    else:
        tip = ("Matching **any** word finds far more documents, so every server has one. Try **All of my words**."
               if not match_all and len(set(terms)) > 1 else
               "Try a more specific search, or a different placement in the sidebar.")
        headline = (f"All three found the same **{found:,} documents{when}**. Every server could have a match, "
                    f"so hybrid had to ask all {n}, the same as distributed. {tip}")
    headline = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", headline)
    st.markdown(f"<p class='headline'>{headline}</p>", unsafe_allow_html=True)

    hits = matches_per_node(terms, months, match_all)
    hybrid_asked = set(runs["hybrid"]["asked"])
    not_needed = set(runs["hybrid"]["skipped_by_score"])
    cols = st.columns(3)
    for col, mode in zip(cols, ["hybrid", "distributed", "centralized"]):
        r, info = runs[mode], MODES[mode]
        with col:
            st.markdown(f"<p class='mode-title' style='color:{info['color']}'>{info['name']}</p>"
                        f"<p class='mode-how'>{info['how']}</p>", unsafe_allow_html=True)
            asked_set = hybrid_asked if mode == "hybrid" else set(system.nodes)
            st.markdown(journey_svg(mode, asked_set, hits, found,
                                    not_needed if mode == "hybrid" else frozenset()), unsafe_allow_html=True)
            servers = "1 server" if mode == "centralized" else f"{r['nodes_queried']} of {n} servers"
            st.markdown(f"<p class='mode-count'>Asked {servers}</p>", unsafe_allow_html=True)

    st.markdown("### What it found")
    results = runs["hybrid"]["results"]
    if not results:
        st.write("Nothing matched. Try fewer or different words.")
        return
    best = results[0].score
    term_set = set(terms)
    for i, r in enumerate(results, 1):
        doc = system.get_document(r.doc_id)
        topic = topic_label(doc.metadata.get("category", ""))
        if "title" in doc.metadata:
            body = (f"<p class='doc'><a href='{html.escape(doc.metadata['url'])}' target='_blank'>"
                    f"{highlight(doc.metadata['title'], term_set)}</a></p>"
                    f"<p class='why'>{snippet(doc.metadata['abstract'], term_set)}</p>")
        else:
            body = f"<p class='doc'>{highlight(doc.content, term_set)}</p>"
        st.markdown(
            f"<div class='card'><div class='rank'>{i}</div><div>"
            f"{body}"
            f"<p class='why'>{explain(doc, terms)}</p>"
            f"<div class='meter' title='Match strength'><span style='width:{100 * r.score / best:.0f}%'></span></div>"
            f"<p class='why' style='margin-top:0.35rem'>{month_name(doc.metadata['month'])} · "
            f"stored on {server_name(r.node_id)}"
            f"<span class='chip'>{html.escape(topic)}</span></p>"
            f"</div></div>", unsafe_allow_html=True)
    if found > len(results):
        st.caption(f"Showing the best {len(results)} of {found:,} matches. The bar shows how strong each match is "
                   f"compared with the best one.")
    else:
        st.caption("The bar shows how strong each match is compared with the best one.")


# ── Page: How it works ────────────────────────────────────────────────────────

def page_how():
    st.title("How it works")
    st.markdown("Think of it like a library system with several branches.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### <span style='color:{MODES['centralized']['color']}'>Centralized</span>", unsafe_allow_html=True)
        st.markdown("**One giant library** with every book. Easy to search, but it gets slow and crowded "
                    "as the collection grows, and if it closes, everything is gone.")
    with c2:
        st.markdown(f"#### <span style='color:{MODES['distributed']['color']}'>Distributed</span>", unsafe_allow_html=True)
        st.markdown("**Several branch libraries.** It scales, but for every question you phone **every** "
                    "branch, even the ones that don't have anything on your topic.")
    with c3:
        st.markdown(f"#### <span style='color:{MODES['hybrid']['color']}'>Hybrid</span>", unsafe_allow_html=True)
        st.markdown("**Branch libraries plus a catalogue** that says which branches have which subjects. "
                    "You check the catalogue first, then phone only the branches that can help.")

    st.markdown("---")
    st.markdown("### What happens when you search, step by step")
    st.markdown(
        "1. **Your words are cleaned up.** “Consistent Hashing!” becomes `consistent`, `hashing`.\n"
        "2. **The directory is checked.** It's a small list of which words live on which servers. "
        "It doesn't store the documents, so it stays small and fast.\n"
        "3. **Only the useful servers are asked.** The directory also knows each server's best possible "
        "score, so after asking the most promising server it skips any server that couldn't beat the "
        "results it already has. The rest are asked at the same time.\n"
        "4. **Each server scores its own matches.** Documents with more of your words score higher, "
        "and rare words count more than common ones.\n"
        "5. **The answers are merged** into one ranked list.")

    example = "consistent hashing"
    terms = tokenize(example)
    st.markdown(f"### What the directory says about “{example}”")
    for t in terms:
        where = sorted(system.metadata.term_to_nodes.get(t, set()))
        names = ", ".join(server_name(n) for n in where) or "no server"
        st.markdown(f"- **{t}** → {names}")
    skipped = len(system.nodes) - len(system.metadata.route_query(terms))
    st.markdown(f"So hybrid can skip **{skipped} of {len(system.nodes)}** servers for this search.")

    st.markdown("---")
    st.markdown("### Why placement matters")
    st.markdown(
        "When a new document arrives, something has to decide which server it goes on. "
        "The directory can only skip a server if it knows that server can't have a match, "
        "so this choice decides how much hybrid can save.")
    st.markdown(
        "| Placement | How documents are placed | What it's good for |\n"
        "| --- | --- | --- |\n"
        "| **Random** | Dealt out like cards, so every server gets the same amount | Even load, but every topic "
        "and every year ends up on every server, so there's little to skip |\n"
        "| **By topic** | Each topic gets its own server: machine learning here, databases there | "
        "Word searches: “graph neural networks” only needs the machine learning server |\n"
        "| **By date** | Each server holds a range of years, like encyclopedia volumes A–F, G–M | "
        "Searches limited to certain years: “2021 only” needs just the server holding 2021 |")
    st.markdown(
        "No placement is best for everything: the right one depends on how people search. "
        "Change **How documents are placed** in the sidebar and search again to see the difference, "
        "or open **What happens at scale**.")


# ── Page: What happens at scale ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def scale_experiment(num_docs: int) -> dict:
    """Index the same generated documents under each placement and count the servers
    asked, for plain word searches and for word searches limited to one year."""
    docs = [Document(d["id"], d["content"], {"category": d["category"], "month": d["month"]})
            for d in generate_dataset(num_docs, seed=42)]
    vocab = [t for topic in TOPICS.values() for t in topic["terms"]]
    rng = random.Random(1)
    queries = rng.sample(vocab, 60)
    years = [rng.randint(2015, 2024) for _ in queries]
    sizes = [2, 4, 8, 16]
    out = {"sizes": sizes, "identical": True,
           "words": {"distributed": sizes}, "dated": {"distributed": sizes}}
    for strat in ["hash", "topic", "range"]:
        out["words"][strat], out["dated"][strat] = [], []
        for n in sizes:
            s = HybridIndexingSystem.build(docs, num_nodes=n, strategy=strat)
            words, dated = [], []
            for q, y in zip(queries, years):
                h = s.hybrid_search(q, top_k=5)
                words.append(h["nodes_queried"])
                dated.append(s.hybrid_search(q, top_k=5, months=years_to_months(y, y))["nodes_queried"])
                if n == 8:
                    c = s.centralized_search(q, top_k=5)
                    if [r.doc_id for r in h["results"]] != [r.doc_id for r in c["results"]]:
                        out["identical"] = False
            out["words"][strat].append(round(statistics.mean(words), 1))
            out["dated"][strat].append(round(statistics.mean(dated), 1))
    return out


LINES = [("distributed", "Distributed (asks every server)", MODES["distributed"]["color"], "dash"),
         ("hash", "Hybrid, placed randomly", "#9ca3af", "solid"),
         ("topic", "Hybrid, grouped by topic", MODES["hybrid"]["color"], "solid"),
         ("range", "Hybrid, grouped by date", "#2563eb", "solid")]


def scale_chart(r: dict, kind: str, title: str):
    fig = go.Figure()
    for key, name, color, dash in LINES:
        fig.add_trace(go.Scatter(x=r["sizes"], y=r[kind][key], name=name, mode="lines+markers",
                                 line=dict(color=color, width=3, dash=dash)))
    fig.update_layout(
        title=title, xaxis=dict(title="Servers in the system", tickvals=r["sizes"]),
        yaxis=dict(title="Servers asked per search", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.28), height=420, margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor="#f3f4f6")
    fig.update_yaxes(gridcolor="#f3f4f6")
    st.plotly_chart(fig, width="stretch")


def page_scale():
    st.title("What happens at scale")
    st.markdown("Twenty documents is a toy. Here the system is rebuilt with thousands of documents "
                "and up to 16 servers, and the same 60 searches are run each time.")
    num_docs = st.select_slider("How many documents", options=[500, 1000, 2000, 5000], value=2000)
    with st.spinner("Building the indexes and running the searches…"):
        r = scale_experiment(num_docs)

    i16 = r["sizes"].index(16)
    w, d = r["words"], r["dated"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<p class='big-stat' style='color:{MODES['distributed']['color']}'>16</p>"
                f"<p class='big-label'>servers asked by <b>distributed</b> for every search, with 16 servers</p>",
                unsafe_allow_html=True)
    c2.markdown(f"<p class='big-stat' style='color:{MODES['hybrid']['color']}'>{w['topic'][i16]:g}</p>"
                f"<p class='big-label'>asked by hybrid for a <b>word search</b> when documents are "
                f"<b>grouped by topic</b></p>", unsafe_allow_html=True)
    c3.markdown(f"<p class='big-stat' style='color:#2563eb'>{d['range'][i16]:g}</p>"
                f"<p class='big-label'>asked by hybrid for a <b>search limited to one year</b> when documents are "
                f"<b>grouped by date</b></p>", unsafe_allow_html=True)

    st.markdown("### 1. Searching by words")
    scale_chart(r, "words", "Word searches: servers asked as the system grows (lower is better)")
    st.markdown("### 2. Searching by words within one year")
    scale_chart(r, "dated", "Searches limited to one year: servers asked (lower is better)")

    st.markdown("### What this shows")
    st.markdown(
        f"- **Distributed** asks every server for every search, so its work grows with the system.\n"
        f"- **Grouping by topic** wins for word searches: about {w['topic'][i16]:g} of 16 servers, "
        f"because each topic lives on its own server.\n"
        f"- **Grouping by date** wins when a search is limited to certain years: about {d['range'][i16]:g} of 16, "
        f"because the directory knows which servers hold those years. For plain word searches it helps "
        f"no more than random placement ({w['range'][i16]:g} vs {w['hash'][i16]:g}), since every topic appears every year.\n"
        f"- **Random placement** is never the best at skipping servers, but it always keeps them evenly loaded.\n\n"
        f"The lesson: **store data the way people search it.**")
    if r["identical"]:
        st.success("Check passed: hybrid and centralized returned exactly the same results for every test search.")
    st.caption("Speed isn't shown in milliseconds here: on one laptop every search takes well under a millisecond. "
               "On a real network, each server asked costs a round trip, so fewer servers asked means faster, "
               "cheaper searches. The documents are generated from templates with example dates, "
               "so treat this as a demonstration.")


# ── Page: Look inside the servers ─────────────────────────────────────────────

def page_inside():
    st.title("Look inside the servers")
    counts = {(f"{server_name(nid)}<br>{year_span(nid)}" if system.strategy == "range" else server_name(nid)):
              len(node.documents) for nid, node in system.nodes.items()}
    fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=MODES["hybrid"]["color"],
                           text=list(counts.values()), textposition="outside"))
    fig.update_layout(title="Documents on each server", height=300, plot_bgcolor="#ffffff",
                      yaxis=dict(visible=False), margin=dict(l=20, r=20, t=50, b=30))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        choice = st.selectbox("Pick a server to see what it holds", list(system.nodes), format_func=server_name)
        node = system.nodes[choice]
        topics = sorted({topic_label(d.metadata.get("category", "")) for d in node.documents.values()})
        st.markdown(f"**{server_name(choice)}** holds {len(node.documents):,} documents "
                    f"about: {', '.join(topics)}.")
        shown = list(node.documents.values())[:25]
        for doc in shown:
            text = (f"<a href='{html.escape(doc.metadata['url'])}' target='_blank'>{html.escape(doc.metadata['title'])}</a>"
                    if "title" in doc.metadata else html.escape(doc.content))
            st.markdown(f"<div class='card' style='grid-template-columns:1fr'><div>"
                        f"<p class='doc'>{text}</p>"
                        f"<span class='chip' style='margin-left:0'>{html.escape(topic_label(doc.metadata.get('category', '')))}</span>"
                        f"</div></div>", unsafe_allow_html=True)
        if len(node.documents) > len(shown):
            st.caption(f"Showing the first {len(shown)} of {len(node.documents):,}.")
    with right:
        st.markdown("#### Ask the directory")
        word = st.text_input("Which servers have this word?", placeholder="e.g. hashing, b-tree, search")
        if word:
            terms = tokenize(word)
            if not terms:
                st.write("That's a filler word the index ignores. Try another.")
            for t in terms:
                where = sorted(system.metadata.term_to_nodes.get(t, set()))
                if where:
                    st.markdown(f"**{t}** is on {', '.join(server_name(n) for n in where)}.")
                else:
                    st.markdown(f"**{t}** isn't on any server.")


{"Try a search": page_search, "How it works": page_how,
 "What happens at scale": page_scale, "Look inside the servers": page_inside}[page]()
