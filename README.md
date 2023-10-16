# DKGE
# may take around 5-10 minutes
!pip install ogb
!python -c "import torch; print(torch.__version__)"
!python -c "import torch; print(torch.version.cuda)"
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
!pip install -q git+https://github.com/snap-stanford/deepsnap.git

import numpy as np
import ogb
import os
import pdb
import random
import torch
import torch_geometric
import tqdm
from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset
from torch.utils.data import DataLoader, Dataset
