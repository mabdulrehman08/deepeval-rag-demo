import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["CONFIDENT_TRACE_ENABLED"] = "false"
from pathlib import Path
from deepeval.metrics import BaseMetric, FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from judge import HeuristicJudge, OfflineTemplate, tokens

ROOT = Path(__file__).parent
DOCS = (ROOT / "corpus.txt").read_text().splitlines()

def retrieve(question):
    return sorted(DOCS, key=lambda d: -len(tokens(question) & tokens(d)))[:1]

def answer(context):
    return context[0].split(": ", 1)[1]

class RetrievalHit(BaseMetric):
    threshold = 1
    def measure(self, test_case, **kwargs):
        self.score = float(any(test_case.expected_output in d for d in test_case.retrieval_context))
        self.success = self.score >= self.threshold
        self.reason = "Expected fact present in retrieved document."
        return self.score
    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)
    def is_successful(self):
        return self.success
    @property
    def __name__(self):
        return "Retrieval hit"

def run():
    rows = []
    for name, question, expected, fault in [
        ("baseline", "What is the deposit?", "The deposit is 500 dollars.", None),
        ("hallucination", "What is the deposit?", "The deposit is 500 dollars.", "answer"),
        ("irrelevant retrieval", "What is the deposit?", "The deposit is 500 dollars.", "retrieval"),
    ]:
        context = retrieve(question)
        if fault == "retrieval":
            context = [DOCS[6]]
        output = "The deposit is 9999 dollars." if fault == "answer" else answer(context)
        case = LLMTestCase(input=question, actual_output=output, expected_output=expected, retrieval_context=context)
        metrics = [RetrievalHit(), FaithfulnessMetric(model=HeuristicJudge(), evaluation_template=OfflineTemplate, async_mode=False), AnswerRelevancyMetric(model=HeuristicJudge(), evaluation_template=OfflineTemplate, async_mode=False)]
        scores = [m.measure(case) for m in metrics]
        rows.append((name, scores))
    return rows

if __name__ == "__main__":
    rows = run()
    table = "| Scenario | Retrieval hit | Faithfulness | Answer relevancy |\n|---|---:|---:|---:|\n"
    table += "".join("| " + name + " | " + " | ".join(f"{s:.2f}" for s in scores) + " |\n" for name, scores in rows)
    (ROOT / "RESULTS.md").write_text(table, encoding="ascii")
    print(table)
