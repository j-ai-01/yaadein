from yaadein.llm import TextGenerator
from yaadein.types import Candidate


class CannedGenerator:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_canned_generator_satisfies_protocol():
    gen: TextGenerator = CannedGenerator("hello")
    assert gen.generate("anything") == "hello"


def test_candidate_holds_extraction_fields():
    c = Candidate(
        content="User prefers pytest",
        category="preference",
        scope="user",
        confidence=0.9,
        evidence_quote="I prefer pytest",
    )
    assert c.category == "preference"
    assert c.scope == "user"
