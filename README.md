# Free Form RL Training

## Start

``` git clone https://github.com/zli12321/long_form_rl.git''

```cd long_form_rl/OpenRLHF

pip install -e .

pip install qa-metrics
```

## Training

```cd ..

cd scripts/no-cot

./grpo_preferenceBert.sh
```

### Notes:

In the grpo_preferenceBert.sh files, remember to change a few things:

- working_dir
- remote_rm_url
- save_path
- use_wandb



## Evaluation
- ToDO: If you want to evaluate, use our template to prompt GPT-4o to generate scores twice then take the average# long_form_rl
