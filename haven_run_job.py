"""
  Minimal code for launching commands on cluster
"""
import os

account_id = os.environ["EAI_ACCOUNT_ID"]

# For saving Cached Files
if account_id == '75ce4cee-6829-4274-80e1-77e89559ddfb':
    # Issam's account
    os.environ['HF_HOME'] = '/mnt/home/cached/'
    os.environ['TORCH_HOME'] = '/mnt/home/cached/'

from haven import haven_jobs as hjb


if __name__ == "__main__":
    # Choose Job Scheduler
    job_config = {
      "account_id": account_id,
      "image": "registry.console.elementai.com/snow.colab/cuda",
      "data": [
          "snow.issam.home:/mnt/home",
          "snow.colab.public:/mnt/public",
          "snow.colab_public.data:/mnt/colab_public",
      ],
      "restartable": True,
      "resources": {"cpu": 4, "mem": 20, "gpu": 4, "gpu_mem": 12,
                     "gpu_model": "!A100"},
      "interactive": False,
      "bid": 9999,
  }

    savedir_base = os.path.abspath("../results/haven_jobs")

    # create a multiline string 
    command = """
    /mnt/home/miniconda38/bin/python train_net.py
    --config-file configs/ril/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
    --num-gpus 4 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0001
    """

    # convert multiline to single line
    command = " ".join(command.split())

    # This command copies a snapshot of the code, saves the logs and errors, 
    # keeps track of the job status, keeps backup, and ensures one unique command per job
    job = hjb.launch_job(
        command, savedir_base=savedir_base, job_scheduler="toolkit",
          job_config=job_config, reset=True
    )