"""Schema-aware custom DeepEval model plus transparent offline prompt templates."""
import json
import re
from deepeval.models import DeepEvalBaseLLM

STOP = set("the is are a an what how much does it cost for".split())
def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP

def packet(kind, **data):
    return json.dumps(dict(kind=kind, **data))

class OfflineTemplate:
    @staticmethod
    def generate_truths(retrieval_context, **kwargs):
        return packet("truths", text=retrieval_context)
    @staticmethod
    def generate_claims(actual_output, **kwargs):
        return packet("claims", text=actual_output)
    @staticmethod
    def generate_statements(actual_output, **kwargs):
        return packet("statements", text=actual_output)
    @staticmethod
    def generate_verdicts(**kwargs):
        return packet("verdicts", **kwargs)
    @staticmethod
    def generate_reason(**kwargs):
        return packet("reason", **kwargs)

class HeuristicJudge(DeepEvalBaseLLM):
    def load_model(self):
        return self
    def get_model_name(self):
        return "offline-token-heuristic"
    def generate(self, prompt, schema=None, **kwargs):
        data = json.loads(prompt)
        kind = data.pop("kind")
        if kind in ("truths", "claims", "statements"):
            text = data["text"]
            result = {kind: text if isinstance(text, list) else [text]}
        elif kind == "reason":
            result = {"reason": "Offline token containment/overlap heuristic; not semantic judgment."}
        else:
            faithful = "claims" in data
            items = data["claims"] if faithful else data["statements"]
            reference = data.get("retrieval_context", data.get("input", ""))
            if isinstance(reference, list):
                reference = " ".join(reference)
            ref = tokens(reference)
            result = {"verdicts": [{"verdict": "yes" if (tokens(s) <= ref if faithful else bool(tokens(s) & ref)) else "no", "reason": "Token support check."} for s in items]}
        return schema.model_validate(result) if schema else json.dumps(result)
    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema, **kwargs)
