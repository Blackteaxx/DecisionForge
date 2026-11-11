"""Chat transport with the training-free baseline protocol."""

from decisionforge.eval.adapters.jev import JevAdapter
from decisionforge.training_free.chat_baseline.processing import (
    ChatProcessing,
)
from decisionforge.training_free.chat_baseline.processing import (
    json_content_equal as json_content_equal,
)


class ChatAdapter(ChatProcessing, JevAdapter):
    pass
