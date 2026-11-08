"""The step-by-step lesson on a run's explanation page, built from the run's own data.

Step 1 picks one question from the run, Step 2 works through the six formulas
for it, Step 3 sums up, Step 4 shows a question where every retrieved chunk was
wrong, and Step 5 walks through fixing it. Every number comes from the stored
results or from the metric functions; none is typed in by hand.

Formulas and calculations are lists of "parts" that the template renders:
{"t": text}, {"frac": (numerator, denominator)} or {"sum": (lower, upper)}.
"""

from collections import Counter

from ..metrics import compute_query_metrics
from ..metrics.ndcg import dcg_terms, ideal_ranking
from ..models import RetrievedChunk
from ..services import builtin_retrievers as br
from ..services.retrieval_adapter import BuiltinRetriever, RetrieverError
from .run_explanation import dec, pct

BUILTIN = ("bm25", "tfidf", "fusion")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def T(text) -> dict:
    return {"t": str(text)}


def F(numerator, denominator) -> dict:
    return {"frac": (str(numerator), str(denominator))}


def S(lower, upper) -> dict:
    return {"sum": (str(lower), str(upper))}


def short(value: float) -> str:
    """1.0 -> '1', 0.5 -> '0.5', 0.6309 -> '0.631', 0 -> '0'."""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def num(value: float) -> str:
    """Like short(), but a score between 0 and 1 keeps two decimals: 0.4 -> '0.40'."""
    text = short(value)
    if 0 < abs(value) < 1 and len(text.split(".")[-1]) == 1:
        text += "0"
    return text


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def id_list(ids: list[str]) -> str:
    if len(ids) <= 1:
        return "".join(ids)
    return ", ".join(ids[:-1]) + " and " + ids[-1]


def be(ids: list[str], present: str = "is") -> str:
    """'is'/'are' (or 'was'/'were') to agree with a list of chunk IDs."""
    plural_form = {"is": "are", "was": "were"}[present]
    return present if len(ids) == 1 else plural_form


def _sum_text(values: list[float], fmt=short, group_over: int = 6) -> str:
    """'1 + 0.5 + 0' for a few values; '13 × 1 + 1 × 0' for many."""
    if len(values) <= group_over:
        return " + ".join(fmt(v) for v in values)
    counts = Counter(round(v, 3) for v in values)
    return " + ".join(f"{n} × {fmt(v)}" for v, n in sorted(counts.items(), key=lambda kv: -kv[0]))


def _chunk(chunks: dict, chunk_id: str) -> dict:
    c = chunks.get(chunk_id)
    return {"title": c.title if c else "", "text": c.text if c else ""}


# ---------------------------------------------------------------------------
# Step 1: one question from the run
# ---------------------------------------------------------------------------
def pick_example(results: list):
    """The question that shows the most: a hit, with missed and irrelevant chunks and an imperfect order."""
    pool = [r for r in results if r.hit_rate == 1] or results

    def richness(r):
        return 4 * (r.relevant_missed > 0) + 2 * (r.irrelevant_found > 0) + (r.ndcg < 0.9995)

    best = max(richness(r) for r in pool)
    return min((r for r in pool if richness(r) == best), key=lambda r: r.ndcg)


def _relevant_expected(result) -> list[tuple[str, int]]:
    return sorted(((cid, g) for cid, g in result.expected.items() if g >= 1), key=lambda kv: (-kv[1], kv[0]))


def _query_data(run, result, chunks: dict) -> dict:
    k = run.top_k
    retrieved = [rc for rc in result.retrieved.all()]
    ranked_ids = [rc.chunk_id for rc in retrieved]
    top = [rc for rc in retrieved if rc.within_k]
    rank_of = {rc.chunk_id: rc.rank for rc in top}
    expected = _relevant_expected(result)
    return {
        "result": result,
        "ranked_ids": ranked_ids,
        "expected_rows": [
            {"chunk_id": cid, "grade": g, **_chunk(chunks, cid), "rank": rank_of.get(cid)} for cid, g in expected
        ],
        "retrieved_rows": [
            {"rank": rc.rank, "chunk_id": rc.chunk_id, "grade": rc.grade, "relevant": rc.is_relevant,
             **_chunk(chunks, rc.chunk_id)}
            for rc in top
        ],
        "total_relevant": result.total_relevant,
        "returned": len(top),
        "found": result.relevant_found,
        "missed_ids": [cid for cid, _ in expected if cid not in rank_of],
        "k": k,
    }


# ---------------------------------------------------------------------------
# Step 2: the six calculations
# ---------------------------------------------------------------------------
def _ndcg_parts(terms) -> tuple[list[dict], str]:
    """'g/log₂(r+1) + 0 + …' up to the last rank that adds anything, and the values as text."""
    last = max((i for i, t in enumerate(terms) if t.gain > 0), default=-1)
    if last < 0:
        return [T("0")], "0"
    parts, values = [], []
    for i, t in enumerate(terms[: last + 1]):
        if i:
            parts.append(T("+"))
        if t.gain > 0:
            parts.append(F(short(t.gain), f"log₂({t.rank + 1})"))
            values.append(short(t.contribution))
        else:
            parts.append(T("0"))
            values.append("0")
    return parts, " + ".join(values)


def _ndcg_block(run, q: dict, chunks: dict) -> dict:
    result, k = q["result"], q["k"]
    grades = {cid: g for cid, g in result.expected.items()}
    relevant_grades = {g for g in grades.values() if g >= 1}
    binary = len(relevant_grades) == 1
    if binary:
        # Every correct chunk has the same grade, so NDCG equals the simple 1/0 version.
        used = {cid: 1 for cid, g in grades.items() if g >= 1}
        gain_type = "linear"
    else:
        used, gain_type = grades, run.ndcg_gain
    terms = dcg_terms(q["ranked_ids"], used, k, gain_type)
    ideal = dcg_terms(ideal_ranking(used), used, k, gain_type)
    dcg = sum(t.contribution for t in terms)
    idcg = sum(t.contribution for t in ideal)
    ndcg = dcg / idcg if idcg else 0.0

    def equation(lhs, terms_):
        parts, values = _ndcg_parts(terms_)
        value = sum(t.contribution for t in terms_)
        out = [T(lhs)] + parts
        if "+" in values:
            out.append(T(f"= {values}"))
        if value:
            out.append(T(f"= {short(value)}"))
        return out

    if binary:
        intro = ("For simple binary relevance: relevant chunk = 1, irrelevant chunk = 0. "
                 f"(All correct chunks for this question have the same grade, {relevant_grades.pop()}, "
                 "so this gives exactly the same NDCG as the graded version.)")
        gain_label = "relᵢ"
    else:
        gain_rule = ("gain = 2^grade − 1 (grade 3 → 7, grade 2 → 3, grade 1 → 1, not relevant → 0)"
                     if gain_type == "exponential" else "gain = grade (not relevant → 0)")
        intro = f"The correct chunks for this question have different grades, so each one counts by its grade: {gain_rule}."
        gain_label = "gainᵢ"

    found_rows = [row for row in q["retrieved_rows"] if row["relevant"]]
    positions = [row["rank"] for row in found_rows]
    best = min(len(ideal), k)
    if ndcg >= 0.9995:
        meaning = "The ranking is perfect: the most relevant chunks are at the top."
    elif not positions:
        meaning = f"No correct chunk is in the top {k}, so DCG and NDCG are 0."
    else:
        meaning = "Our ranking is not perfect."
        gaps = positions != list(range(1, len(positions) + 1))
        if gaps:
            meaning += (f" The correct chunks are at rank{'s' if len(positions) > 1 else ''} "
                        f"{id_list([str(p) for p in positions])}; the ideal order puts the "
                        f"{plural(best, 'best chunk')} at rank{'s 1–' + str(best) if best > 1 else ' 1'}.")
        inversion = next(((a, b) for i, a in enumerate(found_rows) for b in found_rows[i + 1:]
                          if a["grade"] < b["grade"]), None)
        if inversion and not binary:
            a, b = inversion
            meaning += (f" {a['chunk_id']} (grade {a['grade']}) is at rank {a['rank']}, above "
                        f"{b['chunk_id']} (grade {b['grade']}) at rank {b['rank']}: the higher grade should come first.")
        if q["missed_ids"]:
            meaning += f" {id_list(q['missed_ids'])} {be(q['missed_ids'])} missing, which also lowers NDCG."
        if gaps:
            meaning += " We should bring the relevant chunks closer to the top."

    return {
        "intro": intro,
        "table": [
            {"rank": t.rank, "chunk_id": t.chunk_id, "title": _chunk(chunks, t.chunk_id)["title"], "score": short(t.gain)}
            for t in terms
        ],
        "formula": [
            [T(f"NDCG@{k} ="), F(f"DCG@{k}", f"IDCG@{k}")],
            [T(f"DCG@{k} ="), S("i=1", k), F(gain_label, "log₂(i + 1)")],
        ],
        "calc": [
            {"caption": "Actual ranking:", "parts": equation("DCG =", terms)},
            {"caption": "Ideal ranking (correct chunks at the top):", "parts": equation("IDCG =", ideal)},
            {"caption": "Therefore:", "parts": [T("NDCG ="), F(short(dcg), short(idcg)), T(f"= {dec(ndcg)}")]},
        ],
        "dcg": dcg,
        "idcg": idcg,
        "ndcg": ndcg,
        "meaning": meaning,
        "summary_calc": f"{short(dcg)} / {short(idcg)}",
    }


def build_steps(run, q: dict, results: list, chunks: dict) -> tuple[list[dict], list[dict]]:
    result, k = q["result"], q["k"]
    found, total, returned = q["found"], q["total_relevant"], q["returned"]
    p, r, f1 = result.precision, result.recall, result.f1
    n = len(results)
    hits = [x for x in results if x.hit_rate == 1]
    hr = len(hits) / n
    rrs = [x.reciprocal_rank for x in results]
    mrr = sum(rrs) / n

    # Precision
    if found == k:
        p_meaning = f"All {k} retrieved chunks are correct."
    elif found == 0:
        p_meaning = f"None of the {k} retrieved chunks is correct."
    else:
        p_meaning = f"{'Only ' if p < 0.5 else ''}{found} out of {k} retrieved chunks are correct."
    if 0 < found == total < k:
        p_meaning += (f" That is the best this question allows: it has only {plural(total, 'correct chunk')}, "
                      f"so {total}/{k} = {pct(total / k)} is the maximum.")
    p_note = (f"The retriever returned only {returned} chunks, but Precision still divides by K = {k}."
              if returned < k else "")

    # Recall
    missed = q["missed_ids"]
    if not missed:
        r_meaning = f"We found all {plural(total, 'correct chunk')}."
    elif found == 0:
        r_meaning = f"We found none of the {plural(total, 'correct chunk')}."
    else:
        r_meaning = (f"We found {found} out of {total} correct chunks. "
                     f"{'One correct chunk is' if len(missed) == 1 else f'{len(missed)} correct chunks are'} "
                     f"missing ({id_list(missed)}).")

    # F1
    if p + r == 0:
        f1_calc = [T("= 2 ×"), F("0 × 0", "0 + 0"), T("= 0 (defined as 0)")]
        f1_meaning = "Precision and Recall are both 0, so F1 is defined as 0."
    else:
        f1_calc = [T("= 2 ×"), F(f"{num(p)} × {num(r)}", f"{num(p)} + {num(r)}"), T(f"= {num(f1)}")]
        f1_meaning = "It combines Precision and Recall into one score."
        if abs(p - r) < 1e-9:
            f1_meaning += " Precision and Recall are equal here, so F1 is the same value."
        else:
            low = "Precision" if p < r else "Recall"
            f1_meaning += f" It sits closer to the lower one ({low}, {pct(min(p, r))}): both must be high for F1 to be high."

    # Hit rate and MRR over all questions
    query_rows = [
        {"result": x, "found": x.hit_rate == 1, "rank": x.first_relevant_rank,
         "rr": f"1/{x.first_relevant_rank} = {short(x.reciprocal_rank)}" if x.first_relevant_rank else "0",
         "is_example": x.pk == result.pk}
        for x in results
    ]
    rank_counts = Counter(x.first_relevant_rank for x in hits)
    parts = [plural(rank_counts[rank], "question") + f" at rank {rank}" for rank in sorted(rank_counts)]
    misses = n - len(hits)
    if misses:
        parts.append(f"{misses} found none")
    m_meaning = ("We want the first correct result to appear as close to rank 1 as possible. "
                 f"In this run: {', '.join(parts)}.")

    ndcg = _ndcg_block(run, q, chunks)

    return [
        {
            "key": "precision", "label": f"Precision@{k}",
            "question": f"Out of {k} retrieved chunks, how many are correct?",
            "formula": [[T(f"Precision@{k} ="), F("Relevant retrieved", f"Total retrieved (K = {k})")]],
            "note": p_note,
            "calc": [{"caption": "", "parts": [T("="), F(found, k), T(f"= {num(p)}")]}],
            "result": pct(p), "meaning": p_meaning, "summary_calc": f"{found} / {k}",
        },
        {
            "key": "recall", "label": f"Recall@{k}",
            "question": f"Out of {plural(total, 'correct chunk')}, how many did we find?",
            "formula": [[T(f"Recall@{k} ="), F("Relevant retrieved", "Total relevant")]],
            "calc": [{"caption": "", "parts": [T("="), F(found, total), T(f"= {num(r)}")]}],
            "result": pct(r), "meaning": r_meaning, "summary_calc": f"{found} / {total}",
        },
        {
            "key": "f1", "label": "F1 Score",
            "question": "What is the balance between Precision and Recall?",
            "formula": [[T("F1 = 2 ×"), F("P × R", "P + R")]],
            "calc": [{"caption": "", "parts": f1_calc}],
            "result": pct(f1), "meaning": f1_meaning,
            "summary_calc": f"2 × ({num(p)} × {num(r)}) / ({num(p)} + {num(r)})" if p + r else "2 × (0 × 0) / (0 + 0)",
        },
        {
            "key": "hit_rate", "label": f"Hit Rate@{k}",
            "question": "Did we retrieve at least one correct chunk?",
            "intro": f"Hit Rate is counted over many questions. This run tested {plural(n, 'question')}:",
            "table_kind": "hit",
            "formula": [[T(f"Hit Rate@{k} ="), F("Questions with a hit", "Total questions")]],
            "calc": [{"caption": "", "parts": [T("="), F(len(hits), n), T(f"= {pct(hr)}")]}],
            "result": pct(hr),
            "meaning": f"{len(hits)} out of {plural(n, 'question')} found at least one correct chunk.",
            "summary_calc": f"{len(hits)} / {plural(n, 'question')}", "whole_run": True,
        },
        {
            "key": "mrr", "label": "MRR (Mean Reciprocal Rank)",
            "question": "At which position did we find the first correct chunk?",
            "intro": f"Using the same {plural(n, 'question')}:",
            "table_kind": "mrr",
            "formula": [[T("MRR ="), F("1", "N"), S("i=1", "N"), F("1", "rankᵢ")]],
            "calc": [{"caption": "", "parts": [T("="), F(_sum_text(rrs), n), T(f"= {num(mrr)}")]}],
            "result": f"{num(mrr)} ({pct(mrr)})", "meaning": m_meaning,
            "summary_calc": f"({_sum_text(rrs)}) / {n}", "whole_run": True,
        },
        {
            "key": "ndcg", "label": f"NDCG@{k}",
            "question": "Are the most relevant chunks appearing at the top?",
            "intro": ndcg["intro"], "table_kind": "ndcg", "ndcg_table": ndcg["table"],
            "formula": ndcg["formula"], "calc": ndcg["calc"],
            "result": f"{dec(ndcg['ndcg'])} ({pct(ndcg['ndcg'])})", "meaning": ndcg["meaning"],
            "summary_calc": ndcg["summary_calc"],
        },
    ], query_rows


# ---------------------------------------------------------------------------
# Steps 4 and 5: a question where every retrieved chunk was wrong
# ---------------------------------------------------------------------------
FAILURE_POINTS = (
    ("missing_document", "Document not uploaded", "Re-upload the document"),
    ("not_extracted", "Text not extracted", "Fix PDF / OCR parsing"),
    ("chunking", "Correct text missing from chunks", "Fix chunking"),
    ("not_indexed", "Correct chunk not indexed", "Rebuild the index"),
    ("query_mismatch", "Query does not match", "Improve query rewriting"),
    ("ranked_low", "Correct chunk retrieved but ranked low", "Tune the reranker"),
)


def _keywords(text: str) -> list[str]:
    seen, out = set(), []
    for token in br.tokenize(text):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def diagnose(run, result, chunks: dict) -> dict:
    """Checks the app can do itself: is the chunk there, does it share words with the question, how far down is it."""
    expected = [cid for cid, _ in _relevant_expected(result)]
    if not chunks:
        return {"available": False, "checks": [], "likely": None, "suggest": []}
    q_words = _keywords(result.query_text)
    q_stems = {br.light_stem(w) for w in q_words}
    checks, likely = [], None

    missing = [cid for cid in expected if cid not in chunks]
    present = [cid for cid in expected if cid in chunks]
    if missing:
        checks.append({"ok": False, "label": "Document",
                       "text": f"{id_list(missing)} {'is' if len(missing) == 1 else 'are'} not in the dataset's chunk corpus: the source document may be missing."})
        likely = "missing_document"
    else:
        checks.append({"ok": True, "label": "Document", "text": f"{id_list(expected)} {'is' if len(expected) == 1 else 'are'} in the dataset's chunk corpus."})

    empty = [cid for cid in present if not chunks[cid].text.strip()]
    if empty:
        checks.append({"ok": False, "label": "Text", "text": f"{id_list(empty)} has no text: extraction may have failed."})
        likely = likely or "not_extracted"
    elif present:
        checks.append({"ok": True, "label": "Text", "text": "Every expected chunk has text."})

    any_shared = False
    title_words, text_words = [], []
    for cid in present:
        c = chunks[cid]
        words = _keywords(f"{c.title} {c.text}")
        stems = {br.light_stem(x) for x in words}
        shared = [w for w in q_words if br.light_stem(w) in stems]
        any_shared = any_shared or bool(shared)
        new_words = [w for w in words if br.light_stem(w) not in q_stems]
        title_words += [w for w in _keywords(c.title) if br.light_stem(w) not in q_stems and w not in title_words]
        text_words += [w for w, _ in Counter(br.tokenize(c.text)).most_common()
                       if br.light_stem(w) not in q_stems and w not in text_words]
        if shared:
            checks.append({"ok": True, "label": "Keywords",
                           "text": f"The question and {cid} share {plural(len(shared), 'keyword')}: {', '.join(shared)}."})
        else:
            checks.append({"ok": False, "label": "Keywords",
                           "text": f"The question and {cid} share no keywords. Question: {', '.join(q_words) or '—'}. "
                                   f"{cid} uses: {', '.join(new_words[:6])}. Keyword search (BM25) cannot connect them."})
    suggest = title_words[:4]
    for w in text_words:
        if len(suggest) >= 2:
            break
        if w not in suggest:
            suggest.append(w)
    if present and not any_shared:
        likely = likely or "query_mismatch"

    if run.retriever_type in BUILTIN and present:
        try:
            adapter = BuiltinRetriever(run.retriever_type, run.retriever_config)
            adapter.prepare(run.dataset)
            ranking = adapter.rank_all(result.query_text)
        except RetrieverError:
            ranking = None
        if ranking is not None:
            position = {item.chunk_id: i for i, item in enumerate(ranking, start=1)}
            name = run.get_retriever_type_display()
            for cid in present:
                if cid in position:
                    checks.append({"ok": False, "label": "Ranking",
                                   "text": f"Re-run now with the same {name} settings: {cid} comes at rank {position[cid]} "
                                           f"of {len(ranking)} matched chunks, below the top {run.top_k}."})
                    likely = likely or "ranked_low"
                else:
                    checks.append({"ok": False, "label": "Ranking",
                                   "text": f"Re-run now with the same {name} settings: {cid} is never matched "
                                           f"(score 0 for this question), so no K would find it."})
    return {"available": True, "checks": checks, "likely": likely, "suggest": suggest}


def build_failure(run, results: list, chunks: dict, example: dict, fail_pk: int | None) -> dict:
    k = run.top_k
    misses = [r for r in results if r.hit_rate == 0]
    hypothetical = not misses
    if hypothetical:
        q = example
        before = compute_query_metrics([f"__wrong_{i}" for i in range(k)], q["result"].expected, k, run.ndcg_gain)
    else:
        result = next((r for r in misses if r.pk == fail_pk), misses[0])
        q = _query_data(run, result, chunks)
        before = result
    result = q["result"]
    expected = q["expected_rows"]
    grades = {cid: g for cid, g in result.expected.items()}
    ideal_ids = ideal_ranking(grades)
    idcg = sum(t.contribution for t in dcg_terms(ideal_ids, grades, k, run.ndcg_gain))
    total = q["total_relevant"]

    zero_rows = [
        {"label": f"Precision@{k}", "calc": f"0 / {k}", "result": pct(before.precision)},
        {"label": f"Recall@{k}", "calc": f"0 / {total}", "result": pct(before.recall)},
        {"label": "F1 Score", "calc": "2 × (0 × 0) / (0 + 0)", "result": pct(before.f1)},
        {"label": f"Hit Rate@{k}", "calc": "0 / 1 question", "result": pct(before.hit_rate)},
        {"label": "MRR", "calc": "0 / 1 question", "result": short(before.reciprocal_rank)},
        {"label": f"NDCG@{k}", "calc": f"0 / {short(idcg)} (ideal DCG)", "result": short(before.ndcg)},
    ]

    # "After fixing": the metric functions on the best possible top K for this question.
    m = min(len(ideal_ids), k)
    after = compute_query_metrics(ideal_ids[:k] + [f"__other_{i}" for i in range(k - m)], grades, k, run.ndcg_gain)
    before_after = [
        {"label": f"Precision@{k}", "before": pct(before.precision), "after": pct(after.precision)},
        {"label": f"Recall@{k}", "before": pct(before.recall), "after": pct(after.recall)},
        {"label": "F1 Score", "before": pct(before.f1), "after": pct(after.f1)},
        {"label": f"Hit Rate@{k}", "before": pct(before.hit_rate), "after": pct(after.hit_rate)},
        {"label": "MRR", "before": num(before.reciprocal_rank), "after": num(after.reciprocal_rank)},
        {"label": f"NDCG@{k}", "before": short(before.ndcg), "after": short(after.ndcg)},
    ]
    after_note = (
        "After-fixing values are an illustrative example, assuming "
        + ("the relevant chunk is retrieved at rank 1" if m == 1 else
           f"the {m} relevant chunks are retrieved in the first {m} positions, best first")
        + (f", and {plural(k - m, 'irrelevant chunk')} follow{'s' if k - m == 1 else ''}." if k > m else ".")
        + (f" Precision cannot go above {pct(m / k)} for this question, because it has only {plural(total, 'correct chunk')}."
           if m < k else "")
    )

    diag = diagnose(run, result, chunks) if not hypothetical else {"available": False, "checks": [], "likely": None, "suggest": []}
    ids = [row["chunk_id"] for row in expected]
    first = expected[0] if expected else {"chunk_id": "", "title": "", "text": ""}
    q_stems = {br.light_stem(w) for w in _keywords(result.query_text)}
    words = diag["suggest"] or [w for w in _keywords(first["title"] or "") if br.light_stem(w) not in q_stems][:3]
    rewrite = f"{result.query_text.rstrip('?')} + {', '.join(words)}" if words else result.query_text
    snippet = (first["text"][:70] + ("…" if len(first["text"]) > 70 else "")) if first["text"] else ""
    title = first["title"] or first["chunk_id"]

    fix_steps = [
        ("Document validation", "Check whether the correct document exists", f"Is the document behind {first['chunk_id']} (“{title}”) uploaded?"),
        ("Parsing validation", "Check the extracted text",
         f"Is “{snippet}” extracted from the source file?" if snippet else f"Is the text of {first['chunk_id']} extracted correctly?"),
        ("Chunking validation", "Check the document chunks", f"Is the answer stored whole in {id_list(ids)}, not split or cut?"),
        ("Embedding validation", "Check embedding generation", f"{be(ids, 'was').capitalize()} {id_list(ids)} converted into a vector?"),
        ("Index validation", "Check the vector and keyword indexes", f"{be(ids).capitalize()} {id_list(ids)} in the vector index and the BM25 index?"),
        ("Query rewriting", "Improve the user's question", f"“{rewrite}”"),
        ("Hybrid retrieval", "Search with keyword and semantic search",
         f"Search for the words “{', '.join(words)}” and for similar meanings" if words else "Search for both the words and their meaning"),
        ("Reranking", "Reorder the retrieved results", f"Move {id_list(ids)} to the top"),
        ("Evaluation", "Run the same test again", f"Check whether {result.query_external_id} now retrieves {id_list(ids)}"),
    ]
    return {
        "hypothetical": hypothetical,
        "q": q,
        "wrong_count": k if hypothetical else q["returned"],
        "others": [r for r in misses if r.pk != result.pk],
        "zero_rows": zero_rows,
        "before_after": before_after,
        "after_note": after_note,
        "diagnosis": diag,
        "failure_points": [
            {"code": code, "point": point, "action": action, "likely": code == diag["likely"]}
            for code, point, action in FAILURE_POINTS
        ],
        "likely": next(((p, a) for c, p, a in FAILURE_POINTS if c == diag["likely"]), None),
        "fix_steps": [{"n": i, "step": s, "what": w, "example": e} for i, (s, w, e) in enumerate(fix_steps, start=1)],
        "rewrite": rewrite,
        "words": words,
    }


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def build_lesson(run, example_pk: int | None = None, fail_pk: int | None = None) -> dict:
    results = [r for r in run.results.all() if not r.error]
    if not results:
        return {"available": False}
    by_pk = {r.pk: r for r in results}
    chunk_ids = {cid for r in results for cid in r.expected} | set(
        RetrievedChunk.objects.filter(result__run=run, within_k=True).values_list("chunk_id", flat=True)
    )
    chunks = {c.chunk_id: c for c in run.dataset.chunks.filter(chunk_id__in=chunk_ids)}
    has_corpus = run.dataset.chunks.exists()

    example_result = by_pk.get(example_pk) or pick_example(results)
    example = _query_data(run, example_result, chunks)
    steps, query_rows = build_steps(run, example, results, chunks)
    averages = run.averages()
    run_values = {
        "precision": pct(averages["precision"]), "recall": pct(averages["recall"]), "f1": pct(averages["f1"]),
        "hit_rate": pct(averages["hit_rate"]), "mrr": dec(averages["mrr"]), "ndcg": dec(averages["ndcg"]),
    }
    summary_rows = [
        {"label": s["label"].split(" (")[0], "calc": s["summary_calc"], "result": s["result"].split(" (")[0],
         "run": run_values[s["key"]], "whole_run": s.get("whole_run", False)}
        for s in steps
    ]
    return {
        "available": True,
        "k": run.top_k,
        "n": len(results),
        "example": example,
        "choices": results,
        "has_corpus": has_corpus,
        "steps": steps,
        "query_rows": query_rows,
        "summary_rows": summary_rows,
        "failure": build_failure(run, results, chunks, example, fail_pk),
    }

