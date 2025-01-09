
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

## Results

The script outputs two CSV files containing the experimental results:

*   **`experiment_episodes.csv`:** Contains the performance metrics for each episode (or every 10 episodes). This includes the average reward achieved by each method over the last 10 episodes.
*   **`experiment_overall.csv`:** Contains the overall performance statistics for each method across all runs, including the mean and standard deviation of the rewards.

Additionally, the script generates a plot (for the last run) comparing the average rewards of different methods over the episodes.

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

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, new prediction or allocation algorithms, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the  MIT License.
