REGISTRY = {}

from .rnn_agent import RNNAgent
from .macro_agent import MacroAgent
from .value_agent import VALUEAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["macro"] = MacroAgent
REGISTRY["value"] = VALUEAgent