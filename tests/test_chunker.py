from citerx.ingest.chunker import chunk_file
from citerx.config import ROOT

GUIDE = """# Drug X Guideline

## 4.1 Dosage

Drug X is used for condition Y.

WARNING: Do not give Drug X if the patient has renal impairment.

Dosage: 500 mg twice daily for adults.

## 4.2 Monitoring

Check renal function every 6 months.
"""


def test_warning_travels_with_the_dosage(chunk):
    chunks = chunk(GUIDE, max_chars=60)  # tiny limit: packing would normally split these
    dose = next(c for c in chunks if "500 mg" in c.text)
    assert "renal impairment" in dose.text
    assert dose.has_warning


def test_chunks_never_cross_sections(chunk):
    chunks = chunk(GUIDE)
    assert {c.section for c in chunks} == {"4.1 Dosage", "4.2 Monitoring"}
    monitoring = [c for c in chunks if c.section == "4.2 Monitoring"]
    assert all("500 mg" not in c.text for c in monitoring)


def test_heading_styles_detected(chunk):
    text = "DISCHARGE SUMMARY\nPatient notes.\n\n3 Lifestyle Measures\nEat well.\n\nFollow-Up\nSee GP in 1 week."
    assert [c.section for c in chunk(text, min_chars=0)] == ["Discharge Summary", "3 Lifestyle Measures", "Follow-Up"]


def test_long_sections_split_on_sentence_boundaries(chunk):
    body = " ".join(f"Sentence number {i} describes a finding." for i in range(60))
    chunks = chunk(f"## Findings\n\n{body}", max_chars=300)
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_samples_chunk_with_provenance():
    chunks = chunk_file(ROOT / "samples" / "type2_diabetes_protocol.md")
    assert all(c.source == "type2_diabetes_protocol.md" and c.page == 1 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert any(c.has_warning and "lactic acidosis" in c.text for c in chunks)
