from yaadein.gates import apply_gates
from yaadein.types import Candidate

TRANSCRIPT = "USER: I prefer pytest over unittest because less boilerplate\nASSISTANT: Noted."


def cand(**overrides):
    base = dict(
        content="User prefers pytest over unittest",
        category="preference", scope="user", confidence=0.9,
        evidence_quote="I prefer pytest over unittest",
    )
    base.update(overrides)
    return Candidate(**base)


def test_grounded_valid_candidate_survives():
    assert len(apply_gates([cand()], TRANSCRIPT)) == 1


def test_hallucinated_evidence_rejected():
    c = cand(evidence_quote="I love unittest dearly")
    assert apply_gates([c], TRANSCRIPT) == []


def test_grounding_is_whitespace_and_case_insensitive():
    c = cand(evidence_quote="i PREFER pytest   over unittest")
    assert len(apply_gates([c], TRANSCRIPT)) == 1


def test_invalid_category_rejected():
    assert apply_gates([cand(category="opinion")], TRANSCRIPT) == []


def test_low_confidence_rejected():
    assert apply_gates([cand(confidence=0.3)], TRANSCRIPT) == []


def test_too_short_content_rejected():
    assert apply_gates([cand(content="pytest")], TRANSCRIPT) == []


def test_batch_deduped_on_normalized_content():
    dupes = [cand(), cand(content="user prefers PYTEST over unittest")]
    assert len(apply_gates(dupes, TRANSCRIPT)) == 1


def test_budget_keeps_highest_confidence():
    many = [
        cand(content=f"Distinct durable fact number {i} about pytest", confidence=0.6 + i * 0.05,
             evidence_quote="I prefer pytest over unittest")
        for i in range(8)
    ]
    kept = apply_gates(many, TRANSCRIPT)
    assert len(kept) == 5
    assert kept[0].confidence >= kept[-1].confidence
