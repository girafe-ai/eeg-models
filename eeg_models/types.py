from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional

import numpy as np
import torch


Directory = Path
File = Path
npArray = NewType("nupmy array", np.ndarray)
Device = torch.device
