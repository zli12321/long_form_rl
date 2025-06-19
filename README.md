# Free Form RL Training

This repository contains the code and resources for training models on long-form reinforcement learning tasks.

[[📖 Paper](http://arxiv.org/abs/2506.15068)]

## 🚀 Getting Started

To get started with this project, follow the steps below to clone the repository and set up your environment.

**1. Clone the Repository**

```bash
git clone https://github.com/zli12321/long_form_rl.git
cd long_form_rl/OpenRLHF
```

**2. Install Dependencies**

Install the necessary Python packages using pip.

```bash
pip install -e .
pip install qa-metrics
```

## 🏋️ Training

Once the setup is complete, you can begin training the model using the provided scripts.

**1. Navigate to the Training Scripts**

```bash
cd ../scripts/no-cot
```

**2. Configure Your Training Run**

Before launching the training, you must edit the `grpo_preferenceBert.sh` script to match your environment settings.

Open `grpo_preferenceBert.sh` and update the following variables:
* `working_dir`
* `remote_rm_url`
* `save_path`
* `use_wandb`

**3. Run the Training Script**

```bash
./grpo_preferenceBert.sh
```

## 📈 Evaluation

Evaluation procedures are currently under development and will be released soon.

The planned evaluation method involves using our provided template to prompt `GPT-4o` to generate scores twice for each output. The final score will be the average of the two generated scores.

## 📝 Notes

* Please ensure all script paths and configurations are adjusted to fit your specific setup.
* If you encounter any issues or have questions, please feel free to open an issue or submit a pull request on our GitHub repository. We welcome your contributions!

## <a name='citations'></a>Citations

If you find our work helpful for your research, please consider citing our work.   

```
@misc{li2025semanticallyawarerewardsopenendedr1,
      title={Semantically-Aware Rewards for Open-Ended R1 Training in Free-Form Generation}, 
      author={Zongxia Li and Yapei Chang and Yuhang Zhou and Xiyang Wu and Zichao Liang and Yoo Yeon Sung and Jordan Lee Boyd-Graber},
      year={2025},
      eprint={2506.15068},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.15068}, 
}
```