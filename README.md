# Intelligent Go-Explore

[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-1f77b4.svg)](https://openreview.net/forum?id=apErWGzCAA)
[![arXiv](https://img.shields.io/badge/arXiv-2405.15143-b31b1b.svg)](https://arxiv.org/abs/2405.15143)
[![Website](https://img.shields.io/badge/Website-conglu.co.uk-007bff)](https://www.conglu.co.uk/IntelligentGoExplore)
[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/jeffclune/status/1797541076024308135)

<p align="center">
  <img src="figs/ige.png" alt="Intelligent Go-Explore" style="width: 80%;" />
</p>

Intelligent Go-Explore (IGE) integrates the intelligence and internalized human notions of interestingness from giant
pretrained foundation models into all stages of the Go-Explore algorithm, enabling foundation model agents to robustly
explore in complex environments.

**Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models**  
*Cong Lu, Shengran Hu, Jeff Clune. ICLR, 2025.*

---

## Table of Contents

- [Setup](#setup)
- [Running Instructions](#running-instructions)
  - [Notes](#notes)
  - [Game of 24](#game-of-24)
  - [BabyAI-Text](#babyai-text)
  - [TextWorld](#textworld)
- [Citation](#citation)
- [Contact](#contact)

---

## Setup

To install and set up the repository, follow these steps:

1. **Clone the repository and create a Python environment:**

   ```bash
   conda create -n ige python=3.11
   conda activate ige
   pip install -r requirements.txt
   ```

2. **BabyAI-Text Instructions:**

   ```bash
   git clone https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git
   cd Grounding_LLMs_with_online_RL
   pip install blosc
   cd babyai-text/babyai
   pip install -e .
   cd ../gym-minigrid
   pip install -e .
   cd ..
   pip install -e .
   ```

3. **TextWorld Instructions:**

   Make sure you have the necessary system libraries (see [TextWorld GitHub](https://github.com/microsoft/TextWorld)) and then install:

   ```bash
   pip install textworld
   ```

---

## Running Instructions

### Notes

- Each environment is encapsulated in its own file, containing the environment definition, IGE implementation, baselines, and prompts.
- By default, all runs are enabled. Comment out any baselines you do not wish to execute.
- Ensure that the `OPENAI_API_KEY` environment variable is set.
- **Adding a new environment/domain:** We recommend cloning an existing `run.py` file, ensuring the random agent baseline works (which does not require foundation models), and then integrating IGE.

### Game of 24

Run the Game of 24 environment:

```bash
python3 game_of_24/run.py
```

### BabyAI-Text

Run the BabyAI-Text environment:

```bash
python3 babyai/run.py
```

### TextWorld

For TextWorld, several game variants are available:

- **Coin Collector:**

  ```bash
  python3 textworld/run_coin.py
  ```

- **The Cooking Game:**

  ```bash
  python3 textworld/run_cooking.py
  ```

- **Treasure Hunter:**

  ```bash
  python3 textworld/run_treasure.py
  ```

> **Note:** To generate the game files used in TextWorld (located in the `tw_games` directory), the `text_world/misc/make_*.py` scripts were utilized.

---

## Citation

If you find this project useful, please consider citing our work:

```bibtex
@inproceedings{lu2025intelligent,
  title={Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models},
  author={Cong Lu and Shengran Hu and Jeff Clune},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=apErWGzCAA}
}
```

---

## Contact

For any queries, please contact [Cong Lu](mailto:conglu97@outlook.com).
