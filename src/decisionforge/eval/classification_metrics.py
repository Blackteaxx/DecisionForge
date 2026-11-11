"""Compatibility imports; implementations live in decisionforge.benchmarks."""

from decisionforge.benchmarks._classification import (
    accounting as accounting,
)
from decisionforge.benchmarks._classification import (
    applicable as applicable,
)
from decisionforge.benchmarks._classification import (
    label_parameters as label_parameters,
)
from decisionforge.benchmarks._classification import (
    macro_f1 as macro_f1,
)
from decisionforge.benchmarks._classification import (
    native_accuracy as native_accuracy,
)
from decisionforge.benchmarks._classification import (
    native_label as native_label,
)
from decisionforge.benchmarks.banking77.metrics import intent_metrics
from decisionforge.benchmarks.typed_decisions.metrics import (
    kl_parameters as kl_parameters,
)
from decisionforge.benchmarks.typed_decisions.metrics import (
    teacher_metric as teacher_metric,
)
from decisionforge.benchmarks.typed_decisions.metrics import (
    teacher_metrics,
)


def extension_metrics():
    return intent_metrics() + teacher_metrics()
