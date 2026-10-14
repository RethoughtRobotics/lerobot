# VITA on the kinesthetic datasets

| Config | Dataset | Notes |
|---|---|---|
| `pipelines/vita_pose.json` | BlueCupRedContainer_kin_pose | action = measured TCP pose (lead 0), 100k steps |
| `pipelines/vita_lead640ms.json` | BlueCupRedContainer_kin_lead640ms | action = pose 0.64 s ahead (lead 16), 100k steps |
| `pipelines/vita_sweep.json` | SweepTrash_Kin | bimanual, action = next-frame TCP (lead 1), 100k steps |
| `pipelines/vita_pose_aug.json` | BlueCupRedContainer_kin_pose | image augmentation + top camera dropped 30%, 30k steps |

Each config is a `tools/pipeline.py` job: wait for the dataset to be fully uploaded
(`tools/dataset_ready.py`), wait for a GPU, train with `tools/train_vita_hub.sh`, retry on OOM,
upload every checkpoint with `tools/upload_checkpoints.py`. `state_dir` holds its logs and state.

Run one unattended, restarted by cron if the supervisor dies:

    (crontab -l; echo "*/10 * * * * $PWD/tools/pipeline_watchdog.sh $PWD/tasks/vita_kin/pipelines/vita_pose_aug.json") | crontab -

Without a supervisor: `tasks/vita_kin/train_pose_aug_topdrop30.sh [gpu]`.
