"""Plain-language explanation of a run's six scores, built from the run's own numbers.

Everything here is derived from the stored MetricResult rows, so the
explanation always matches the scores on the results page.
"""

from collections import Counter
from dataclasses import dataclass, field

from ..metrics import METRICS_BY_KEY


def pct(value: float) -> str:
    text = f"{value * 100:.1f}"
    return f"{text[:-2] if text.endswith('.0') else text}%"


def dec(value: float) -> str:
    return f"{value:.3f}"


def _grouped_sum(values: list[float], fmt) -> str:
    """'13 × 1.000 + 1 × 0.000' for a list of per-query values (largest first)."""
    counts = Counter(round(v, 3) for v in values)
    return " + ".join(f"{n} × {fmt(v)}" for v, n in sorted(counts.items(), key=lambda kv: -kv[0]))


@dataclass
class MetricStory:
    key: str
    label: str
    value_text: str
    value: float
    meaning: str
    formula: str
    calculation: str
    reading: str
    tone: str  # 'good', 'watch' or 'bad': drives the icon next to the reading
    problem: str
    fixes: str
    extra: dict = field(default_factory=dict)


def explain_run(run) -> dict:
    results = [r for r in run.results.all() if not r.error]
    n, k = len(results), run.top_k
    if not n:
        return {"available": False}
    avg = run.averages()

    slots = n * k
    relevant = sum(r.relevant_found for r in results)
    irrelevant = sum(r.irrelevant_found for r in results)
    empty = slots - relevant - irrelevant
    total_relevant = sum(r.total_relevant for r in results)
    precision_ceiling = sum(min(r.total_relevant, k) for r in results) / slots
    recall_ceiling = sum(min(r.total_relevant, k) / r.total_relevant for r in results) / n
    hits = [r for r in results if r.hit_rate == 1]
    misses = [r for r in results if r.hit_rate == 0]
    full_recall = sum(1 for r in results if r.recall >= 0.9999)
    rank_counts = Counter(r.first_relevant_rank for r in hits)
    rank1 = rank_counts.get(1, 0)
    perfect_ndcg = sum(1 for r in results if r.ndcg >= 0.9995)
    imperfect = sorted((r for r in hits if r.ndcg < 0.9995), key=lambda r: r.ndcg)
    f1_of_averages = (
        2 * avg["precision"] * avg["recall"] / (avg["precision"] + avg["recall"])
        if avg["precision"] + avg["recall"] else 0.0
    )
    label = {key: METRICS_BY_KEY[key].label(k) for key in METRICS_BY_KEY}
    miss_ids = ", ".join(r.query_external_id for r in misses[:6]) + (" …" if len(misses) > 6 else "")

    # ---- Precision -------------------------------------------------------
    p = avg["precision"]
    share_of_best = p / precision_ceiling if precision_ceiling else 0
    if share_of_best >= 0.9:
        p_reading, p_tone = (
            f"Low-looking but close to the best possible ({pct(precision_ceiling)}): the queries have only "
            f"{total_relevant} relevant chunks for {slots} slots, so most slots can never be relevant. "
            "This is the dataset's limit, not noise from the retriever.", "good")
    elif share_of_best >= 0.6:
        p_reading, p_tone = (
            f"Below the best possible ({pct(precision_ceiling)}): some slots that could hold a relevant chunk "
            "hold an irrelevant one.", "watch")
    else:
        p_reading, p_tone = (
            f"Far below the best possible ({pct(precision_ceiling)}): the top {k} is mostly noise.", "bad")

    # ---- Recall ----------------------------------------------------------
    r_avg = avg["recall"]
    r_tone = "good" if r_avg >= 0.9 else "watch" if r_avg >= 0.7 else "bad"
    r_reading = (
        f"{full_recall} of {n} queries found every relevant chunk"
        + (f"; {len(misses)} found none ({miss_ids})." if misses else ".")
    )
    if recall_ceiling < 0.9999:
        r_reading += f" Best possible at K = {k} is {pct(recall_ceiling)}, because some queries have more relevant chunks than K."

    # ---- Hit rate --------------------------------------------------------
    h_tone = "good" if not misses else "bad" if len(misses) / n > 0.2 else "watch"
    h_reading = (
        f"Every query got at least one useful chunk." if not misses else
        f"{len(misses)} quer{'y' if len(misses) == 1 else 'ies'} got nothing useful in the top {k}: {miss_ids}. "
        "For these, the LLM has no correct context and is likely to say \"I don't know\" or make something up."
    )

    # ---- MRR -------------------------------------------------------------
    m = avg["mrr"]
    rr_terms = [f"{rank_counts[rank]} × 1/{rank}" for rank in sorted(rank_counts)]
    if misses:
        rr_terms.append(f"{len(misses)} × 0")
    m_reading = (
        f"{rank1} of {len(hits)} queries that found something had it at rank 1."
        if hits else "No query found a relevant chunk."
    )
    if hits and rank1 == len(hits):
        m_reading += " MRR is below 1 only because of the queries that found nothing."
    m_tone = "good" if hits and rank1 / len(hits) >= 0.9 else "watch" if m >= 0.5 else "bad"

    # ---- NDCG ------------------------------------------------------------
    g = avg["ndcg"]
    g_tone = "good" if g >= 0.85 else "watch" if g >= 0.7 else "bad"
    g_reading = f"{perfect_ndcg} of {n} queries have a perfect order (1.000)."
    if imperfect:
        g_reading += " Not perfect: " + ", ".join(
            f"{r.query_external_id} {dec(r.ndcg)}" for r in imperfect[:5]
        ) + ". In these, a relevant chunk sits below an irrelevant one, or a lower grade sits above a higher grade."
    if g < m - 0.005:
        g_reading += f" NDCG is lower than MRR ({dec(m)}): the first chunk is usually right, the ones after it less so."

    stories = [
        MetricStory(
            "precision", label["precision"], pct(p), p,
            meaning=f"Of the {k} chunks the retriever hands over, how many are actually relevant?",
            formula=f"relevant chunks in the top {k} ÷ {k}",
            calculation=f"{relevant} relevant chunks in {n} × {k} = {slots} slots = {pct(p)}",
            reading=p_reading, tone=p_tone,
            problem="Noise: irrelevant chunks in the context distract the LLM, cost tokens and can lead it to a wrong answer.",
            fixes="Add a reranker, lower K, filter by metadata (department, date), improve chunking.",
            extra={"ceiling": precision_ceiling},
        ),
        MetricStory(
            "recall", label["recall"], pct(r_avg), r_avg,
            meaning="Of all the relevant chunks that exist for a question, how many did the retriever find?",
            formula=f"relevant chunks in the top {k} ÷ all relevant chunks for the query",
            calculation=f"mean of {n} per-query recalls: ({_grouped_sum([r.recall for r in results], dec)}) ÷ {n} = {pct(r_avg)}",
            reading=r_reading, tone=r_tone,
            problem="Missing knowledge: the answer is incomplete, or the LLM fills the gap by guessing.",
            fixes="Hybrid search (keyword + vector), query expansion or rewriting, larger K, better embeddings, chunk overlap.",
        ),
        MetricStory(
            "f1", label["f1"], pct(avg["f1"]), avg["f1"],
            meaning="One number that is only high when precision and recall are both high.",
            formula="2 × Precision × Recall ÷ (Precision + Recall), per query, then averaged",
            calculation=f"mean of {n} per-query F1 scores = {pct(avg['f1'])}",
            reading=(
                (f"Precision is capped by the dataset (see Precision), and the harmonic mean sits close to the "
                 f"smaller value, so F1 stays low. "
                 + ("Recall is strong, so the low F1 is not a retriever problem. " if r_tone == "good"
                    else f"Recall ({pct(r_avg)}) is the part the retriever can improve. ")
                 if p_tone == "good" else
                 f"Pulled down by precision ({pct(p)}), because the harmonic mean sits close to the smaller value. ")
                + f"Note: F1 of the two averages would be {pct(f1_of_averages)}; the average of per-query F1 scores is "
                f"{pct(avg['f1'])}. They are different numbers, and this page uses the second."
            ),
            tone="good" if p_tone == "good" and r_tone == "good" else "watch",
            problem="Hides a trade-off: raising K improves recall but lowers precision. F1 shows whether a change helped overall.",
            fixes="Use it to choose K and reranker cut-offs; look at precision and recall separately to see which side moved.",
        ),
        MetricStory(
            "hit_rate", label["hit_rate"], pct(avg["hit_rate"]), avg["hit_rate"],
            meaning=f"Did the question get at least one useful chunk in the top {k}? Yes = 1, no = 0.",
            formula=f"1 if any of the top {k} chunks is relevant, else 0; averaged over queries",
            calculation=f"{len(hits)} of {n} queries had a hit = {len(hits)} ÷ {n} = {pct(avg['hit_rate'])}",
            reading=h_reading, tone=h_tone,
            problem="Complete failures: the question gets no correct context at all. Users see these as wrong answers.",
            fixes="Fix vocabulary mismatch (synonyms, dense/vector retrieval), add missing documents, check chunking of those topics.",
        ),
        MetricStory(
            "mrr", label["mrr"], dec(m), m,
            meaning="How high up is the first useful chunk? Rank 1 scores 1, rank 2 scores 1/2, rank 3 scores 1/3 …",
            formula="1 ÷ rank of the first relevant chunk (0 if none), averaged over queries",
            calculation=f"({' + '.join(rr_terms)}) ÷ {n} = {dec(m)}",
            reading=m_reading, tone=m_tone,
            problem="Buried answers: the right chunk is there but low, where the LLM pays less attention (\"lost in the middle\").",
            fixes="Add a cross-encoder reranker; tune the hybrid weights so the strongest match comes first.",
        ),
        MetricStory(
            "ndcg", label["ndcg"], dec(g), g,
            meaning="Is the whole top list in the best order, with the most relevant chunks (higher grades) first?",
            formula=f"DCG@{k} ÷ ideal DCG@{k};  DCG = Σ (2^grade − 1) ÷ log₂(rank + 1)",
            calculation=f"mean of {n} per-query NDCG scores = {dec(g)}",
            reading=g_reading, tone=g_tone,
            problem="Poor ordering: a somewhat-relevant chunk ranks above the chunk that really answers the question.",
            fixes="Reranker trained or tuned on graded labels; boost titles or key fields; check the queries listed above.",
        ),
    ]

    # ---- How the six link together, checked on this run ------------------
    rules = [
        {
            "rule": f"Precision ≤ best possible precision ({pct(precision_ceiling)})",
            "why": f"A query with R relevant chunks can fill at most R of the {k} slots.",
            "holds": p <= precision_ceiling + 1e-9,
            "values": f"{pct(p)} ≤ {pct(precision_ceiling)}",
        },
        {
            "rule": "Hit Rate ≥ Recall",
            "why": "If a query finds any relevant chunk its hit is 1, and its recall is at most 1.",
            "holds": avg["hit_rate"] >= r_avg - 1e-9,
            "values": f"{pct(avg['hit_rate'])} ≥ {pct(r_avg)}",
        },
        {
            "rule": "MRR ≤ Hit Rate",
            "why": "A query's reciprocal rank is 1/rank, never more than its hit (1).",
            "holds": m <= avg["hit_rate"] + 1e-9,
            "values": f"{dec(m)} ≤ {dec(avg['hit_rate'])}",
        },
        {
            "rule": "F1 sits between precision and recall, nearer the smaller",
            "why": "The harmonic mean punishes the weaker side.",
            "holds": min(p, r_avg) - 1e-9 <= avg["f1"] <= max(p, r_avg) + 1e-9,
            "values": f"{pct(p)} ≤ {pct(avg['f1'])} ≤ {pct(r_avg)}",
        },
        {
            "rule": "A query with no hit scores 0 on all six",
            "why": "No relevant chunk means nothing to count, rank or weight.",
            "holds": all(r.precision == 0 and r.ndcg == 0 and r.reciprocal_rank == 0 for r in misses),
            "values": f"{len(misses)} such quer{'y' if len(misses) == 1 else 'ies'}",
        },
    ]

    # ---- Overall summary and what to fix first -----------------------------
    summary = (
        f"Out of {n} test questions, the retriever found at least one correct chunk for {len(hits)} "
        f"(Hit Rate {pct(avg['hit_rate'])})"
    )
    summary += (
        f", and when it did, the correct chunk was at the top for {rank1} of them (MRR {dec(m)}). "
        if hits else ". "
    )
    summary += (
        f"It found {pct(r_avg)} of the relevant chunks on average (Recall). Only {pct(p)} of the {k} chunks it "
        f"returns are relevant (Precision), "
        + ("but that is close to the best this dataset allows, " if share_of_best >= 0.9 else "")
        + f"which pulls F1 down to {pct(avg['f1'])}. The order of the results scores {dec(g)} out of 1 (NDCG)."
    )

    priorities = []
    if misses:
        priorities.append(
            f"Fix the {len(misses)} quer{'y' if len(misses) == 1 else 'ies'} with no useful result ({miss_ids}) first: "
            "these give users wrong answers. Open each one to see what came back instead."
        )
    if r_avg < 0.9 and len(misses) < n:
        priorities.append("Recall is the next gap: relevant chunks are being missed. Try hybrid search or query rewriting.")
    if hits and rank1 / len(hits) < 0.9:
        priorities.append("Relevant chunks often appear below rank 1: a reranker is the usual fix.")
    if imperfect and g < 0.9:
        priorities.append("Ordering is weak for some queries (low NDCG): check grades and add a reranker.")
    if share_of_best < 0.9:
        priorities.append(f"Precision is below what is possible ({pct(precision_ceiling)}): cut noise with a reranker or a smaller K.")
    if not priorities:
        priorities.append("No major gap. Add harder test questions (synonyms, multi-part questions) so the dataset keeps catching regressions.")

    return {
        "available": True,
        "n": n,
        "k": k,
        "summary": summary,
        "stories": stories,
        "rules": rules,
        "priorities": priorities,
        "misses": misses,
        "counts": {
            "slots": slots, "relevant": relevant, "irrelevant": irrelevant, "empty": empty,
            "total_relevant": total_relevant, "precision_ceiling": precision_ceiling,
        },
    }
