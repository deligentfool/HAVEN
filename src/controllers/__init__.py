REGISTRY = {}

from .basic_controller import BasicMAC
from .macro_controller import MacroMAC
from .value_controller import ValueMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["macro_mac"] = MacroMAC
REGISTRY["value_mac"] = ValueMAC