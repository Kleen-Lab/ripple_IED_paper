# ripple IED paper code
* First, set up Python environment
### CPU Only
```
conda create -n klab_cpu python=3.9.23 -y
conda activate klab_cpu
python --version
# Should show Python 3.9.23
pip install -r requirements-cpu.txt
```
### Check we have implemeted all packages required
```
python - <<'PY'
import sys
import numpy
import scipy
import pandas
import sklearn
import torch
import torchvision
import transformers

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```
### With GPU (Seahorse environment)
```
conda create -n klab_gpu python=3.9.23 -y
conda activate klab_gpu
python --version
pip install -r requirements-gpu.txt
```
