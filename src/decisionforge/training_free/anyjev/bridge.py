"""Local label-only extension for pinned AnyJev 35311c58955716cb8955ce1368a665988bc63735.

The official Decider still constructs every prompt, scores all K cyclic shifts,
computes the content-free prior and marginalizes with logmean. Only its existing
_labels_for hook changes for Choice K > 26; small questions use upstream verbatim.
"""

from decisionforge.label_tokens import allocate_labels

LARGE_LABEL_INSTRUCTION = (
    "\nEach displayed option label may contain multiple letters (for example AA); "
    "emit the complete displayed label as a single token."
)


def make_question(question_type, kind, **kwargs):
    """Lift only Choice's 26-letter factory cap, retaining its other validation."""
    if kind != "choice" or len(kwargs["options"]) <= 26:
        return getattr(question_type, kind)(**kwargs)
    from anyjev.question import QuestionError

    options = tuple(str(option) for option in kwargs["options"])
    if len(options) < 2:
        raise QuestionError("choice needs at least 2 options")
    if len(set(options)) != len(options):
        raise QuestionError("choice options must be unique")
    # The public dataclass is the factory's own return type. No upstream globals
    # or installed files are changed; all non-size constraints match its factory.
    return question_type(
        "choice", kwargs["text"] + LARGE_LABEL_INSTRUCTION, options, kwargs.get("name")
    )


def make_decider(backend, **settings):
    """Instantiate the official pipeline with a per-instance large-label hook."""
    from anyjev import Decider

    class CatalogDecider(Decider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_protocols = {}

        def _labels_for(self, question):
            if question.kind != "choice" or question.k <= 26:
                return super()._labels_for(question)
            if question.k not in self.label_protocols:
                self.label_protocols[question.k] = allocate_labels(
                    self.backend.tokenizer,
                    question.k,
                    encoding_style="bare_then_space",
                    legacy_count=26,
                )
            protocol = self.label_protocols[question.k]
            return protocol["labels"], protocol["token_ids"]

    return CatalogDecider(backend, **settings)
