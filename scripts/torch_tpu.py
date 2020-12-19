# Installs PyTorch, PyTorch/XLA, and Torchvision
# Copy this cell into your own notebooks to use PyTorch on Cloud TPUs 
# Warning: this may take a couple minutes to run

import collections
from datetime import datetime, timedelta
import os
import requests
import threading

_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
VERSION = "torch_xla==nightly" #@param ["xrt==1.15.0", "torch_xla==nightly"] 
CONFIG = {
    'xrt==1.15.0': _VersionConfig('1.15', '1.15.0'),
    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
        (datetime.today() - timedelta(1)).strftime('%Y%m%d'))),
}[VERSION]
DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TPU_IP_ADDRESS = "10.216.137.218"
XRT_TPU_CONFIG = "tpu_worker;0;$TPU_IP_ADDRESS:8470"
# Update TPU XRT version
def update_server_xrt():
  print('Updating server-side XRT to {} ...'.format(CONFIG.server))
  url = 'http://{TPU_ADDRESS}:8470/requestversion/{XRT_VERSION}'.format(
      TPU_ADDRESS=TPU_IP_ADDRESS,
      XRT_VERSION=CONFIG.server,
  )
  print(url)
  print('Done updating server-side XRT: {}'.format(requests.post(url)))

update = threading.Thread(target=update_server_xrt)
update.start()

# Install Colab TPU compat PyTorch/TPU wheels and dependencies
os.system("pip uninstall -y torch torchvision")
os.system("gsutil cp '$DIST_BUCKET/$TORCH_WHEEL' .")
os.system("gsutil cp '$DIST_BUCKET/$TORCH_XLA_WHEEL' .")
os.system("gsutil cp '$DIST_BUCKET/$TORCHVISION_WHEEL' .")
os.system("pip install '$TORCH_WHEEL'")
os.system("pip install '$TORCH_XLA_WHEEL'")
os.system("pip install '$TORCHVISION_WHEEL'")
os.system("pip install transformers")
os.system("sudo apt-get install libomp5")
update.join()