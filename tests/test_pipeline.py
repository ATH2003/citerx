from citerx.generate import llm as llm_module
from citerx.generate.extractive import NO_ANSWER


def test_extractive_answer_is_cited_and_grounded(rx):
    ans = rx.ask("What are the renal contraindications for metformin?", use_llm=False)
    assert not ans.abstained
    assert "30" in ans.text and "[1]" in ans.text
    assert ans.groundedness == 1.0


def test_out_of_scope_question_abstains(rx):
    ans = rx.ask("Who won the football world cup in 2022?", use_llm=False)
    assert ans.abstained and ans.text == NO_ANSWER


def test_hallucinating_llm_gets_caught(rx, monkeypatch):
    fake = "Metformin is contraindicated when eGFR is below 30 [1]. The maximum dose is 3000 mg per day [1]."
    monkeypatch.setattr("citerx.pipeline.chat", lambda *a, **k: fake)
    ans = rx.ask("What is the maximum dose of metformin?", use_llm=True)
    assert ans.mode == "llm"
    assert [c.status for c in ans.claims] == ["supported", "unsupported"]
    assert ans.claims[1].missing_numbers == ["3000"]


def test_llm_failure_falls_back_to_extractive(rx, monkeypatch):
    def boom(*a, **k):
        raise llm_module.LLMError("connection refused")

    monkeypatch.setattr("citerx.pipeline.chat", boom)
    ans = rx.ask("What is the starting dose of ramipril?", use_llm=True)
    assert ans.mode == "extractive" and "2.5 mg" in ans.text
    assert "fell back" in rx.last_error


def test_reasoning_tags_are_stripped(monkeypatch):
    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = b'{"choices":[{"message":{"content":"<think>hmm</think>Answer [1]."}}]}'
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", lambda *a, **k: Resp(body))
    assert llm_module.chat([], "http://x", "m") == "Answer [1]."


def test_reingesting_replaces_document(rx):
    from citerx.config import ROOT

    before = len(rx.index.chunks)
    rx.ingest(ROOT / "samples" / "hypertension_pathway.txt")
    assert len(rx.index.chunks) == before
