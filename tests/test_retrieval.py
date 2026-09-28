from citerx.index.bm25 import BM25
from citerx.retrieve.fusion import rrf
from citerx.text import numbers, stem


def test_rrf_rewards_agreement():
    fused = rrf({"dense": [1, 2, 3], "bm25": [2, 4, 1]})
    order = [item for item, _, _ in fused]
    assert set(order[:2]) == {1, 2}  # found by both retrievers
    assert order.index(3) > 1 and order.index(4) > 1


def test_bm25_matches_exact_clinical_tokens():
    bm = BM25(["eGFR below 30 mL/min contraindicated", "blood pressure 140/90 mmHg", "general advice on diet"])
    assert bm.top("140/90", 1)[0][0] == 1
    assert bm.top("egfr 30", 1)[0][0] == 0


def test_stemming_joins_word_forms():
    assert stem("discharged") == stem("discharge")
    assert stem("contraindicated") == stem("contraindications")
    assert stem("stopped") == stem("stop")
    assert stem("allergic") == stem("allergies")


def test_numbers_normalised():
    assert numbers("HbA1c 7.0% and 2,5 mg at 140/90") == {"1", "7", "2.5", "140/90"}


def test_safety_expansion_adds_section_warning(rx):
    # "starting dose" should also surface the pregnancy/ACE+ARB warning in that section
    results = rx.retriever.retrieve("What is the starting dose of ramipril?")
    texts = " ".join(r.chunk.text for r in results)
    assert "contraindicated in pregnancy" in texts


def test_retrieval_finds_expected_sections(rx):
    cases = {
        "renal contraindications for metformin": "4.1",
        "blood pressure target over 80": "2 Treatment",
        "ejection fraction on echocardiogram": "Diagnosis",
    }
    for q, section in cases.items():
        top3 = [r.chunk.section for r in rx.retriever.retrieve(q)[:3]]
        assert any(s.startswith(section) for s in top3), (q, top3)
