import subprocess
import sys
import torch

# 1️⃣ Check for GPU
has_gpu = torch.cuda.is_available()
print("GPU detected:", has_gpu)

# 2️⃣ Decide Torch versions
if has_gpu:
    torch_version = "torch==2.5.1+cu124"
    torchaudio_version = "torchaudio==2.5.1+cu124"
    torchvision_version = "torchvision==0.20.1+cu124"
    extra_index = "-f https://download.pytorch.org/whl/torch_stable.html"
else:
    torch_version = "torch==2.5.1"
    torchaudio_version = "torchaudio==2.5.1"
    torchvision_version = "torchvision==0.20.1"
    extra_index = None  # <-- None instead of empty string

# 3️⃣ Install Torch
torch_install_cmd = [sys.executable, "-m", "pip", "install",
                     torch_version, torchaudio_version, torchvision_version]

if extra_index:
    torch_install_cmd.append(extra_index)

subprocess.check_call(torch_install_cmd)

# 4️⃣ Install other pinned libs
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "whisperx==3.3.0",
                       "pyannote.audio==3.3.2",
                       "speechbrain==1.0.1",
                       "tqdm",
                       "numpy",
                       "pandas",
                       "nltk",
                       "matplotlib"
                      ])

# 5️⃣ Install convert-audio with whisperx extra
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[whisperx]"])
