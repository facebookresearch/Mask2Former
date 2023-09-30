import os

account_id = os.environ["EAI_ACCOUNT_ID"]

# For saving Cached Files
if account_id == "75ce4cee-6829-4274-80e1-77e89559ddfb":
    # Issam's account
    os.environ["HF_HOME"] = "/mnt/home/cached/"
    os.environ["TORCH_HOME"] = "/mnt/home/cached/"

from haven import haven_jobs as hjb

# Choose Job Scheduler
JOB_CONFIG = {
    "account_id": account_id,
    "image": "registry.console.elementai.com/snow.colab/cuda",
    "data": [
        "snow.issam.home:/mnt/home",
        "snow.colab.public:/mnt/public",
        "snow.colab_public.data:/mnt/colab_public",
    ],
    "restartable": True,
    "resources": {
        "cpu": 5,
        "mem": 80,
        "gpu": 4,
        "gpu_mem": 16,
        "gpu_model": "!A100",
    },
    "interactive": False,
    "bid": 9999,
}
