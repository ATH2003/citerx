from citerx.verify.claims import cited_ids, verify

from conftest import make_sources

SOURCE = (
    "Metformin is contraindicated when eGFR is below 30 mL/min/1.73m2 because of the risk of lactic acidosis. "
    "Do not exceed 1000 mg per day when eGFR is between 30 and 45."
)
OTHER = "Hypoglycaemia is treated with 15 to 20 g of fast-acting carbohydrate."


def statuses(answer, *sources):
    return [c.status for c in verify(answer, make_sources(*sources))]


def test_faithful_claim_is_supported():
    assert statuses("Metformin is contraindicated when eGFR is below 30 [1].", SOURCE) == ["supported"]


def test_invented_dose_is_flagged_even_with_high_word_overlap():
    checks = verify("Do not exceed 1500 mg per day when eGFR is between 30 and 45 [1].", make_sources(SOURCE))
    assert checks[0].status == "unsupported"
    assert checks[0].missing_numbers == ["1500"]


def test_wrong_citation_is_flagged():
    # True fact, but attributed to the wrong source
    assert statuses("Metformin is contraindicated below an eGFR of 30 [2].", SOURCE, OTHER) == ["unsupported"]


def test_uncited_claim_is_flagged():
    assert statuses("Metformin is safe to use in pregnancy.", SOURCE) == ["uncited"]


def test_citation_formats_and_inheritance():
    assert cited_ids("x [1, 3] and [Source 2]") == [1, 3, 2]
    checks = verify("Metformin carries a risk of lactic acidosis. It is contraindicated below eGFR 30 [1].", make_sources(SOURCE))
    assert [c.cited for c in checks] == [[1], [1]]
    assert all(c.status == "supported" for c in checks)


def test_abstention_is_not_a_claim():
    assert verify("The provided documents do not answer this question.", make_sources(SOURCE)) == []
