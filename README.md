# Local RAG regression lab

A tiny, key-free DeepEval demo: ten bundled rental FAQ documents, keyword-overlap
retrieval, and a deterministic extractive mock answerer. Two injected regressions
separate unsupported generation from irrelevant retrieval.

## Run (Python 3.13)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python demo.py
```

The final command prints and regenerates RESULTS.md. Run the regression suite with
`.venv\Scripts\python -m pytest -q`. Expected failures are asserted as detections,
so the test suite itself passes. Installation needs internet; evaluation needs no
API keys, model downloads, or hosted service.

## What is measured

- Retrieval hit: a custom DeepEval BaseMetric checks the expected fact in context.
- FaithfulnessMetric: real DeepEval claim extraction, verdict aggregation, and scoring
  through its custom model and evaluation_template hooks.
- AnswerRelevancyMetric: real DeepEval statement/verdict scoring through the same hooks.

HeuristicJudge implements DeepEvalBaseLLM and returns schema-validated responses.
OfflineTemplate sends structured JSON instructions to that judge. Faithfulness uses
claim-token containment in retrieved context; relevancy uses non-stopword overlap.
These are deliberately narrow smoke checks, not substitutes for semantic evaluation.
A hallucination can still be relevant; an irrelevant retrieved fact can still be
faithfully copied. RESULTS.md makes those distinctions visible. The keyword retriever
uses corpus order to break ties; it is intentionally small rather than production RAG.

To use a real judge, replace model=HeuristicJudge() with a supported model name
(e.g. model="gpt-4.1-mini"), remove evaluation_template=OfflineTemplate so DeepEval's
normal natural-language prompts are restored, and configure that provider's credentials.
Alternatively implement DeepEvalBaseLLM for a locally hosted model. Never feed these
JSON-only templates to a real judge as a quality benchmark.

This mirrors the retrieval-versus-generation debugging needs of my RAG advisor work
at Xiente, using synthetic rental facts rather than private project data.

API references: https://deepeval.com/docs/metrics-faithfulness and
https://deepeval.com/docs/metrics-answer-relevancy . Pinned to DeepEval 4.3.4;
custom hooks were checked against the installed package source.

## Example output

| Scenario | Retrieval hit | Faithfulness | Answer relevancy |
|---|---:|---:|---:|
| baseline | 1.00 | 1.00 | 1.00 |
| hallucination | 1.00 | 0.00 | 1.00 |
| irrelevant retrieval | 0.00 | 1.00 | 0.00 |
