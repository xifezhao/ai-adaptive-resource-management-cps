
# Adaptive Resource Management System

This repository contains a Python implementation of an adaptive resource management system. The system utilizes various demand prediction techniques and resource allocation strategies, including traditional methods and reinforcement learning approaches, to dynamically manage resources based on predicted demand.

## Overview

The core idea is to intelligently allocate a limited number of resources to meet fluctuating demands. The system is designed with modularity in mind, allowing for easy swapping of different prediction and allocation algorithms.

**Key Components:**

1. **Data Preprocessing:**  Normalizes raw data to prepare it for prediction models.
2. **Demand Prediction:** Implements several time-series forecasting methods to predict future demand. Currently includes:
    *   **LSTM (Long Short-Term Memory) Network:** A type of recurrent neural network effective for sequential data.
    *   **ARIMA (Autoregressive Integrated Moving Average):** A statistical model for time series forecasting.
    *   **Transformer Network:** A neural network architecture that leverages attention mechanisms.
    *   **Random Forest Regressor:** An ensemble learning method for regression tasks.
    *   **Linear Regression:** A simple linear model for prediction.
3. **Resource Allocation:**  Determines how many resources to allocate based on the predicted demand. Implements the following strategies:
    *   **Traditional Methods:**
        *   **Fixed Allocation:**  Allocates a constant number of resources.
        *   **Threshold-Based Allocation:** Allocates resources based on whether the average of recent data exceeds a threshold.
    *   **Reinforcement Learning Methods:**
        *   **Q-Learning:** An off-policy temporal difference learning algorithm.
        *   **SARSA (State-Action-Reward-State-Action):** An on-policy temporal difference learning algorithm.
        *   **Policy Gradient (REINFORCE):**  A method that directly optimizes the policy function.
    *   **Rule-Based Allocation:** Allocates resources based on predefined rules and the predicted demand.
4. **Adaptive Control:** Monitors the system's performance and can trigger adaptations, such as adjusting exploration rates in reinforcement learning agents.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or higher installed. You will also need the following Python packages:

```bash
pip install numpy pandas torch matplotlib statsmodels scikit-learn
```

You can install all the necessary dependencies using the provided `requirements.txt` file (create one if it doesn't exist with the listed packages):

```bash
pip install -r requirements.txt
```

### Running the Experiment

To run the simulation, execute the main Python script:

```bash
IntelligentResourceManager_EN.py  # 
IntelligentResourceManager_CN.py  # 中文注释，对国人友好的注解版本
```

The script will perform multiple runs of the experiment, comparing the performance of different resource management strategies.

### Experiment Setup

The `run_experiment` function in the script defines the parameters for the simulation:

*   `n_resources`: The total number of resources available.
*   `n_actions`: The number of possible resource allocation actions for reinforcement learning agents.
*   `episodes`: The number of simulation steps (episodes) to run.

The script initializes and compares the following resource management approaches:

*   **Traditional:**
    *   Fixed Allocation
    *   Threshold-Based Allocation
*   **Adaptive (combinations of predictors and allocators):**
    *   LSTM Predictor + Q-Learning Allocator
    *   ARIMA Predictor + SARSA Allocator
    *   Transformer Predictor + Policy Gradient Allocator
    *   Random Forest Predictor + Rule-Based Allocator
    *   Linear Regression Predictor + Q-Learning Allocator

## Experiment Data Description

This repository contains the results of resource management experiments stored in two CSV files:

1. `experiment_episodes.csv`  
2. `experiment_overall.csv`  

Below is a detailed explanation of the purpose and structure of each file.

---

### 1. `experiment_episodes.csv`

#### Purpose  
This file contains detailed data for each experiment episode. It tracks the performance of different resource management methods over time, making it useful for dynamic performance analysis.

#### Structure  
The file includes the following columns:

| Column                        | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| `Run ID`                      | The identifier for the experiment run (e.g., `1`, `2`, etc.).                                   |
| `Episode`                     | The episode number within the experiment run.                                                  |
| `Fixed Reward`                | Average reward obtained using the Fixed Allocation method during the episode.                  |
| `Threshold Reward`            | Average reward obtained using the Threshold Allocation method during the episode.              |
| `Adaptive (LSTM + QL) Reward` | Average reward for the adaptive method combining LSTM predictor and Q-Learning allocator.      |
| `Adaptive (ARIMA + SARSA) Reward` | Average reward for the adaptive method combining ARIMA predictor and SARSA allocator.       |
| `Adaptive (Transformer + PG) Reward` | Average reward for the adaptive method combining Transformer predictor and Policy Gradient allocator. |
| `Adaptive (RF + Rule) Reward` | Average reward for the adaptive method combining Random Forest predictor and Rule-Based allocator. |
| `Adaptive (LR + QL) Reward`   | Average reward for the adaptive method combining Linear Regression predictor and Q-Learning allocator. |

#### Example Data  

| Run ID | Episode | Fixed Reward | Threshold Reward | Adaptive (LSTM + QL) Reward | Adaptive (ARIMA + SARSA) Reward | Adaptive (Transformer + PG) Reward | Adaptive (RF + Rule) Reward | Adaptive (LR + QL) Reward |
|--------|---------|--------------|-------------------|-----------------------------|----------------------------------|--------------------------------------|-----------------------------|----------------------------|
| 1      | 0       | 0.50         | 0.45              | 0.62                        | 0.58                             | 0.63                                 | 0.55                        | 0.60                       |
| 1      | 10      | 0.52         | 0.48              | 0.65                        | 0.61                             | 0.68                                 | 0.57                        | 0.64                       |

---

### 2. `experiment_overall.csv`

#### Purpose  
This file summarizes the overall performance of different resource management methods for each experiment run. It provides statistical metrics such as the mean and standard deviation of the rewards.

#### Structure  
The file includes the following columns:

| Column                            | Description                                                                                     |
|-----------------------------------|-------------------------------------------------------------------------------------------------|
| `Run ID`                          | The identifier for the experiment run (e.g., `1`, `2`, etc.).                                   |
| `Fixed Mean`                      | Mean reward across all episodes using the Fixed Allocation method.                              |
| `Fixed Std`                       | Standard deviation of rewards using the Fixed Allocation method.                                |
| `Threshold Mean`                  | Mean reward across all episodes using the Threshold Allocation method.                          |
| `Threshold Std`                   | Standard deviation of rewards using the Threshold Allocation method.                            |
| `Adaptive (LSTM + QL) Mean`       | Mean reward for the adaptive method combining LSTM predictor and Q-Learning allocator.          |
| `Adaptive (LSTM + QL) Std`        | Standard deviation of rewards for the adaptive method combining LSTM predictor and Q-Learning allocator. |
| `Adaptive (ARIMA + SARSA) Mean`   | Mean reward for the adaptive method combining ARIMA predictor and SARSA allocator.              |
| `Adaptive (ARIMA + SARSA) Std`    | Standard deviation of rewards for the adaptive method combining ARIMA predictor and SARSA allocator. |
| `Adaptive (Transformer + PG) Mean` | Mean reward for the adaptive method combining Transformer predictor and Policy Gradient allocator. |
| `Adaptive (Transformer + PG) Std` | Standard deviation of rewards for the adaptive method combining Transformer predictor and Policy Gradient allocator. |
| `Adaptive (RF + Rule) Mean`       | Mean reward for the adaptive method combining Random Forest predictor and Rule-Based allocator.  |
| `Adaptive (RF + Rule) Std`        | Standard deviation of rewards for the adaptive method combining Random Forest predictor and Rule-Based allocator. |
| `Adaptive (LR + QL) Mean`         | Mean reward for the adaptive method combining Linear Regression predictor and Q-Learning allocator. |
| `Adaptive (LR + QL) Std`          | Standard deviation of rewards for the adaptive method combining Linear Regression predictor and Q-Learning allocator. |

#### Example Data  

| Run ID | Fixed Mean | Fixed Std | Threshold Mean | Threshold Std | Adaptive (LSTM + QL) Mean | Adaptive (LSTM + QL) Std | Adaptive (ARIMA + SARSA) Mean | Adaptive (ARIMA + SARSA) Std | ... |
|--------|------------|-----------|----------------|---------------|---------------------------|--------------------------|-------------------------------|--------------------------------|-----|
| 1      | 0.52       | 0.03      | 0.49           | 0.02          | 0.63                      | 0.05                     | 0.60                          | 0.04                           | ... |

---

### Usage  

- **Dynamic Analysis**: Use `experiment_episodes.csv` to visualize how the rewards change over episodes for each method. For example, plot reward curves to assess the convergence and stability of each method.  
- **Summary Comparison**: Use `experiment_overall.csv` to compare the overall performance (mean and variability) of different resource management methods across runs.  


## Code Structure

The code is organized into several classes, each responsible for a specific part of the system:

*   **`IDemandPredictor` and `IResourceAllocator`:**  Interface classes defining the contract for demand predictors and resource allocators, respectively.
*   **`TraditionalResourceManager`:** Implements traditional fixed and threshold-based resource allocation methods.
*   **`DataPreprocessor`:** Handles the preprocessing of raw data.
*   **Demand Predictor Classes (`LSTMPredictor`, `ARIMAPredictor`, `TransformerPredictor`, `RandomForestPredictor`, `LinearRegressionPredictor`):** Implement different demand prediction models.
*   **Resource Allocator Classes (`QLearningAllocator`, `SARSAAllocator`, `PolicyGradientAllocator`, `RuleBasedAllocator`):** Implement various resource allocation strategies.
*   **`AdaptiveController`:** Monitors system performance and triggers adaptations.
*   **`AdaptiveResourceManager`:** The main class that orchestrates the interaction between the different components.
*   **`run_experiment`:**  Sets up and runs a single experimental run.
*   **`main`:**  Executes multiple experimental runs and saves the results.

![Average rewards per run](average_rewards_per_run.pdf)

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, new prediction or allocation algorithms, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the  MIT License.
