
---

# CoSee: Reproducible Collaboration Protocol for Document VQA

CoSee is an auditable shared-workspace protocol and evaluation pipeline for document visual question answering (VQA) under strict small-model budgets. This repository provides the necessary code, models, and evaluation scripts to run the experiments in our paper, focusing on reproducibility and cost-normalized evaluation.

## Table of Contents

- [CoSee: Reproducible Collaboration Protocol for Document VQA](#cosee-reproducible-collaboration-protocol-for-document-vqa)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Dataset Setup](#dataset-setup)
    - [Running Experiments](#running-experiments)
    - [Evaluation and Results](#evaluation-and-results)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Introduction

CoSee introduces a shared visual workspace for document VQA, where agents collaborate by externalizing intermediate notes and evidence. We provide a measurement-oriented protocol that allows systematic evaluation under resource constraints (small models and single-GPU setups).

This repository contains:

* Code for the collaboration protocols (`baseline_single`, `single_board`, `two_qwen`, and `verified_board`).
* Scripts for evaluating models and computing metrics.
* Preprocessing, logging, and integrity auditing tools for reproducibility.
* Configuration files for setting up experiments.

## Getting Started

### Requirements

Before running the code, ensure that you have the following software installed:

* Python 3.7+
* PyTorch (with GPU support)
* Other dependencies: `numpy`, `scipy`, `matplotlib`, `transformers`, `torchmetrics`, etc.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cosee.git
   cd cosee
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) If you use `conda`, you can install dependencies from `requirements-conda.txt`:

   ```bash
   conda create --name cosee-env --file requirements-conda.txt
   conda activate cosee-env
   ```

### Dataset Setup

CoSee uses several public document VQA benchmarks. Ensure you have the following datasets:

* **SlideVQA**: The test set with 600 examples.
* **ChartQAPro**: The test set with 800 examples.
* **VQAonline**: A dataset for webpage screenshot VQA.

You can download these datasets from their respective sources (e.g., [SlideVQA Dataset](https://example.com), [ChartQAPro Dataset](https://example.com)). Once downloaded, place them in the `data/` directory.

### Running Experiments

1. To run the baseline single-turn answering protocol (`baseline_single`), execute the following command:

   ```bash
   python scripts/run_experiment.py --config configs/baseline_single.json
   ```

2. To run the single-agent board protocol (`single_board`):

   ```bash
   python scripts/run_experiment.py --config configs/single_board.json
   ```

3. To run the two-agent scanner-checker protocol (`two_qwen`):

   ```bash
   python scripts/run_experiment.py --config configs/two_qwen.json
   ```

4. To run the verified-board control (`verified_board`):

   ```bash
   python scripts/run_experiment.py --config configs/verified_board.json
   ```

Each experiment will output logs, metrics, and results in the `results/` directory. You can adjust the configurations in the `configs/` folder to modify parameters like the number of tokens per call, the number of steps, or the model backbone.

### Evaluation and Results

After running experiments, you can evaluate the results using the provided scripts. The evaluation includes metrics such as Exact Match (EM), Loose Match, and more. The results will be logged and can be found in the `results/` directory.

To generate plots for results, such as accuracy vs. cost (token generation), you can run:

```bash
python scripts/plot_results.py --input results/ --output results/plots/
```

The resulting plots will help visualize the trade-offs between model performance and computational cost.

## Contributing

We welcome contributions to improve the CoSee project! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* The CoSee project is built upon several existing document VQA benchmarks, including SlideVQA, ChartQAPro, and VQAonline.
* Special thanks to the authors of these benchmarks for providing the datasets used in our evaluation.

---