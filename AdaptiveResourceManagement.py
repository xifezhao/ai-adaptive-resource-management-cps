import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# 忽略 statsmodels 的 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- 预测器接口 ---
class IDemandPredictor:
    def predict(self, data: np.ndarray) -> float:
        raise NotImplementedError

# --- 分配器接口 ---
class IResourceAllocator:
    def select_action(self, state: int) -> int:
        raise NotImplementedError
    def update(self, state: int, action: int, reward: float, next_state: int):
        raise NotImplementedError

# --- 传统资源管理器 (保持不变) ---
class TraditionalResourceManager:
    def __init__(self, n_resources: int, method: str = 'fixed'):
        self.n_resources = n_resources
        self.method = method
        self.threshold = 0.5

    def allocate_resources(self, data: np.ndarray) -> Tuple[int, float]:
        if self.method == 'fixed':
            allocation = self.calculate_fixed_allocation()
            reward = self.calculate_fixed_reward()
        else:  # threshold method
            allocation = self.calculate_threshold_allocation(data)
            reward = self.calculate_threshold_reward(data)
        return allocation, reward

    def calculate_fixed_allocation(self) -> int:
        return self.n_resources // 2

    def calculate_fixed_reward(self) -> float:
        return 0.5

    def calculate_threshold_allocation(self, data: np.ndarray) -> int:
        return self.n_resources if np.mean(data) > self.threshold else 0

    def calculate_threshold_reward(self, data: np.ndarray) -> float:
        return float(np.random.random() < np.mean(data))

# --- 1. 数据预处理模块 (保持不变) ---
class DataPreprocessor:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        normalized_data = (raw_data - np.mean(raw_data)) / (np.std(raw_data) + 1e-8)
        self.data_buffer.append(normalized_data)
        while len(self.data_buffer) < self.window_size:
            self.data_buffer.append(normalized_data)
        return np.array(self.data_buffer).reshape(1, self.window_size, -1)

# --- 2. 需求预测模块 (新增其他方法) ---

# 2.1 LSTM 预测器 (保持不变，但实现接口)
class LSTMPredictor(nn.Module, IDemandPredictor):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def predict(self, data: np.ndarray) -> float:
        with torch.no_grad():
            prediction = self.forward(torch.FloatTensor(data)).item()
        return prediction

# 2.2 ARIMA 预测器
class ARIMAPredictor(IDemandPredictor):
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.history = []

    def predict(self, data: np.ndarray) -> float:
        self.history.extend(data.flatten().tolist())
        if len(self.history) >= self.order[0] + self.order[1]:
            try:
                model = ARIMA(self.history, order=self.order)
                model_fit = model.fit()
                output = model_fit.forecast()
                return output[0]
            except Exception as e:
                print(f"ARIMA prediction error: {e}")
                return 0.5
        else:
            return 0.5

# 2.3 简化的 Transformer 预测器
class TransformerPredictor(nn.Module, IDemandPredictor):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, batch_first=True) # 关键修改: batch_first=True
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.embedding(src) # [batch_size, seq_len, input_size] -> [batch_size, seq_len, d_model]
        output = self.transformer(src, src) # [batch_size, seq_len, d_model]
        output = self.fc(output[:, -1, :]) # 取最后一个时间步的输出 [batch_size, output_size]
        return output

    def predict(self, data: np.ndarray) -> float:
        with torch.no_grad():
            prediction = self.forward(torch.FloatTensor(data)).item()
        return prediction

# 2.4 随机森林预测器
class RandomForestPredictor(IDemandPredictor):
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        self.history = deque(maxlen=50) # 存储最近的数据用于训练

    def predict(self, data: np.ndarray) -> float:
        self.history.extend(data.flatten().tolist())
        if len(self.history) > 1:
            # 使用历史数据训练模型
            X = np.array(range(len(self.history) -1)).reshape(-1, 1)
            y = np.array(list(self.history)[:-1])
            self.model.fit(X, y)
            # 预测下一个时间点的值
            return self.model.predict(np.array([[len(self.history) - 1]]))[0]
        else:
            return 0.5

# 2.5 线性回归预测器
class LinearRegressionPredictor(IDemandPredictor):
    def __init__(self):
        self.model = LinearRegression()
        self.history = deque(maxlen=50)

    def predict(self, data: np.ndarray) -> float:
        self.history.extend(data.flatten().tolist())
        if len(self.history) > 1:
            X = np.array(range(len(self.history) - 1)).reshape(-1, 1)
            y = np.array(list(self.history)[:-1])
            self.model.fit(X, y)
            return self.model.predict(np.array([[len(self.history) - 1]]))[0]
        else:
            return 0.5

# --- 3. 资源分配模块 (新增其他方法) ---

# 3.1 Q-Learning 分配器 (保持不变，但实现接口)
class QLearningAllocator(IResourceAllocator):
    def __init__(self, n_resources: int, n_actions: int):
        self.n_resources = n_resources
        self.n_actions = n_actions
        self.q_table = np.zeros((n_resources, n_actions))
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# 3.2 SARSA 分配器
class SARSAAllocator(IResourceAllocator):
    def __init__(self, n_resources: int, n_actions: int, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        self.n_resources = n_resources
        self.n_actions = n_actions
        self.q_table = np.zeros((n_resources, n_actions))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        next_action = self.select_action(next_state)
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]
        new_value = old_value + self.learning_rate * (reward + self.gamma * next_value - old_value)
        self.q_table[state, action] = new_value

# 3.3 简单的策略梯度分配器 (REINFORCE)
class PolicyGradientAllocator(IResourceAllocator):
    def __init__(self, n_resources: int, n_actions: int, learning_rate=0.01, gamma=0.99):
        self.n_resources = n_resources
        self.n_actions = n_actions
        # 修改网络结构，确保维度匹配
        self.policy_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),  # 输出维度应为 n_actions
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []

    def select_action(self, state: int) -> int:
        # 将整数 state 转换为浮点数张量，并确保维度正确
        state_tensor = torch.tensor([[float(state)]], dtype=torch.float32)
        probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def update(self, state: int, action: int, reward: float, next_state: int):
        self.rewards.append(reward)

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

# 3.4 基于规则的分配器
class RuleBasedAllocator(IResourceAllocator):
    def __init__(self, n_resources: int, threshold: float = 0.7):
        self.n_resources = n_resources
        self.threshold = threshold

    def select_action(self, predicted_demand: float) -> int:
        # 如果预测需求超过阈值，则分配较多资源
        if predicted_demand > self.threshold:
            return self.n_resources // 2 + 1
        else:
            return self.n_resources // 2 - 1

    def update(self, state: int, action: int, reward: float, next_state: int):
        # 基于规则的方法不需要更新
        pass

# --- 4. 自适应控制模块 (保持不变) ---
class AdaptiveController:
    def __init__(self):
        self.performance_history = []
        self.adaptation_threshold = 0.8

    def evaluate_performance(self, metrics: Dict) -> float:
        performance = np.mean([v for v in metrics.values()])
        self.performance_history.append(performance)
        return performance

    def need_adaptation(self) -> bool:
        if len(self.performance_history) < 5:
            return False
        recent_performance = np.mean(self.performance_history[-5:])
        return recent_performance < self.adaptation_threshold

# --- 5. 主框架 ---
class AdaptiveResourceManager:
    def __init__(self, n_resources: int, n_actions: int, predictor: IDemandPredictor, allocator: IResourceAllocator):
        self.preprocessor = DataPreprocessor()
        self.predictor = predictor
        self.allocator = allocator
        self.controller = AdaptiveController()

    def step(self, raw_data: np.ndarray, current_state: int) -> Tuple[int, float]:
        # 数据预处理
        processed_data = self.preprocessor.preprocess(raw_data)

        # 需求预测
        predicted_demand = self.predictor.predict(processed_data)

        # 资源分配
        action = self.allocator.select_action(current_state)
        if isinstance(self.allocator, RuleBasedAllocator):
            action = self.allocator.select_action(predicted_demand) # 基于规则的分配器直接使用预测值

        # 模拟环境反馈
        reward = self.simulate_environment_feedback(action, predicted_demand)
        next_state = (current_state + action) % self.allocator.n_resources

        # 更新分配器 (如果需要)
        if not isinstance(self.allocator, RuleBasedAllocator):
            self.allocator.update(current_state, action, reward, next_state)

        # 性能评估和自适应控制
        metrics = {
            'reward': reward,
            'prediction_accuracy': self.calculate_prediction_accuracy(predicted_demand)
        }

        if self.controller.need_adaptation():
            self.adapt_system()

        return next_state, reward

    def simulate_environment_feedback(self, action: int, predicted_demand: float) -> float:
        if isinstance(self.allocator, RuleBasedAllocator):
            # 对于 RuleBasedAllocator，我们假设 action 就是期望的分配结果
            return random.random() * (1.0 if action > self.allocator.n_resources // 2 else 0.5)
        else:
            target_action = int(predicted_demand * self.allocator.n_actions)
            return random.random() * (1.0 if action == target_action else 0.5)

    def calculate_prediction_accuracy(self, predicted_demand: float) -> float:
        return random.random()

    def adapt_system(self):
        if isinstance(self.allocator, QLearningAllocator):
            self.allocator.epsilon *= 0.95
        elif isinstance(self.allocator, SARSAAllocator):
            self.allocator.epsilon *= 0.95
        self.controller.adaptation_threshold *= 0.99

def main():
    try:
        n_resources = 5
        n_actions = 3
        episodes = 1000

        # 初始化传统管理器
        fixed_manager = TraditionalResourceManager(n_resources=n_resources, method='fixed')
        threshold_manager = TraditionalResourceManager(n_resources=n_resources, method='threshold')

        # 初始化不同的预测器
        lstm_predictor = LSTMPredictor(input_size=1, hidden_size=32, output_size=1)
        arima_predictor = ARIMAPredictor()
        transformer_predictor = TransformerPredictor(input_size=1, d_model=32, nhead=4, num_layers=2, output_size=1)
        rf_predictor = RandomForestPredictor()
        lr_predictor = LinearRegressionPredictor()

        # 初始化不同的分配器
        q_learning_allocator = QLearningAllocator(n_resources, n_actions)
        sarsa_allocator = SARSAAllocator(n_resources, n_actions)
        policy_gradient_allocator = PolicyGradientAllocator(n_resources, n_actions)
        rule_based_allocator = RuleBasedAllocator(n_resources)

        # 初始化自适应管理器
        adaptive_manager_lstm_ql = AdaptiveResourceManager(n_resources, n_actions, predictor=lstm_predictor, allocator=q_learning_allocator)
        adaptive_manager_arima_sarsa = AdaptiveResourceManager(n_resources, n_actions, predictor=arima_predictor, allocator=sarsa_allocator)
        adaptive_manager_transformer_pg = AdaptiveResourceManager(n_resources, n_actions, predictor=transformer_predictor, allocator=policy_gradient_allocator)
        adaptive_manager_rf_rule = AdaptiveResourceManager(n_resources, n_actions, predictor=rf_predictor, allocator=rule_based_allocator)
        adaptive_manager_lr_ql = AdaptiveResourceManager(n_resources, n_actions, predictor=lr_predictor, allocator=q_learning_allocator)

        # 记录所有方法的性能
        fixed_rewards = []
        threshold_rewards = []
        adaptive_rewards_lstm_ql = []
        adaptive_rewards_arima_sarsa = []
        adaptive_rewards_transformer_pg = []
        adaptive_rewards_rf_rule = []
        adaptive_rewards_lr_ql = []

        # 模拟运行
        current_state_lstm_ql = 0
        current_state_arima_sarsa = 0
        current_state_transformer_pg = 0
        current_state_rf_rule = 0
        current_state_lr_ql = 0

        for episode in range(episodes):
            # 生成模拟数据
            raw_data = np.random.randn(1)

            # 传统方法
            _, fixed_reward = fixed_manager.allocate_resources(raw_data)
            fixed_rewards.append(fixed_reward)

            _, threshold_reward = threshold_manager.allocate_resources(raw_data)
            threshold_rewards.append(threshold_reward)

            # 自适应方法 - LSTM + Q-Learning
            next_state_lstm_ql, adaptive_reward_lstm_ql = adaptive_manager_lstm_ql.step(raw_data, current_state_lstm_ql)
            adaptive_rewards_lstm_ql.append(adaptive_reward_lstm_ql)
            current_state_lstm_ql = next_state_lstm_ql
            if episode > 0 and episode % 10 == 0 and isinstance(adaptive_manager_lstm_ql.allocator, PolicyGradientAllocator):
                adaptive_manager_lstm_ql.allocator.finish_episode()

            # 自适应方法 - ARIMA + SARSA
            next_state_arima_sarsa, adaptive_reward_arima_sarsa = adaptive_manager_arima_sarsa.step(raw_data, current_state_arima_sarsa)
            adaptive_rewards_arima_sarsa.append(adaptive_reward_arima_sarsa)
            current_state_arima_sarsa = next_state_arima_sarsa
            if episode > 0 and episode % 10 == 0 and isinstance(adaptive_manager_arima_sarsa.allocator, PolicyGradientAllocator):
                adaptive_manager_arima_sarsa.allocator.finish_episode()

            # 自适应方法 - Transformer + Policy Gradient
            next_state_transformer_pg, adaptive_reward_transformer_pg = adaptive_manager_transformer_pg.step(raw_data, current_state_transformer_pg)
            adaptive_rewards_transformer_pg.append(adaptive_reward_transformer_pg)
            current_state_transformer_pg = next_state_transformer_pg
            if episode > 0 and episode % 10 == 0 and isinstance(adaptive_manager_transformer_pg.allocator, PolicyGradientAllocator):
                adaptive_manager_transformer_pg.allocator.finish_episode()

            # 自适应方法 - RF + Rule Based
            predicted_demand_rf = adaptive_manager_rf_rule.predictor.predict(adaptive_manager_rf_rule.preprocessor.preprocess(raw_data))
            action_rf = adaptive_manager_rf_rule.allocator.select_action(predicted_demand_rf)
            reward_rf = adaptive_manager_rf_rule.simulate_environment_feedback(action_rf, predicted_demand_rf)
            adaptive_rewards_rf_rule.append(reward_rf)
            # RuleBasedAllocator doesn't update state in the same way
            current_state_rf_rule = (current_state_rf_rule + action_rf) % n_resources

            # 自适应方法 - LR + Q-Learning
            next_state_lr_ql, adaptive_reward_lr_ql = adaptive_manager_lr_ql.step(raw_data, current_state_lr_ql)
            adaptive_rewards_lr_ql.append(adaptive_reward_lr_ql)
            current_state_lr_ql = next_state_lr_ql
            if episode > 0 and episode % 10 == 0 and isinstance(adaptive_manager_lr_ql.allocator, PolicyGradientAllocator):
                adaptive_manager_lr_ql.allocator.finish_episode()

            if episode % 10 == 0:
                print(f"\nEpisode {episode}:")
                print(f"Fixed Method - Average Reward: {np.mean(fixed_rewards[-10:]):.3f}")
                print(f"Threshold Method - Average Reward: {np.mean(threshold_rewards[-10:]):.3f}")
                print(f"Adaptive (LSTM + QL) - Average Reward: {np.mean(adaptive_rewards_lstm_ql[-10:]):.3f}")
                print(f"Adaptive (ARIMA + SARSA) - Average Reward: {np.mean(adaptive_rewards_arima_sarsa[-10:]):.3f}")
                print(f"Adaptive (Transformer + PG) - Average Reward: {np.mean(adaptive_rewards_transformer_pg[-10:]):.3f}")
                print(f"Adaptive (RF + Rule) - Average Reward: {np.mean(adaptive_rewards_rf_rule[-10:]):.3f}")
                print(f"Adaptive (LR + QL) - Average Reward: {np.mean(adaptive_rewards_lr_ql[-10:]):.3f}")

        # 计算总体性能统计
        print("\nOverall Performance:")
        print(f"Fixed Method - Mean: {np.mean(fixed_rewards):.3f}, Std: {np.std(fixed_rewards):.3f}")
        print(f"Threshold Method - Mean: {np.mean(threshold_rewards):.3f}, Std: {np.std(threshold_rewards):.3f}")
        print(f"Adaptive (LSTM + QL) - Mean: {np.mean(adaptive_rewards_lstm_ql):.3f}, Std: {np.std(adaptive_rewards_lstm_ql):.3f}")
        print(f"Adaptive (ARIMA + SARSA) - Mean: {np.mean(adaptive_rewards_arima_sarsa):.3f}, Std: {np.std(adaptive_rewards_arima_sarsa):.3f}")
        print(f"Adaptive (Transformer + PG) - Mean: {np.mean(adaptive_rewards_transformer_pg):.3f}, Std: {np.std(adaptive_rewards_transformer_pg):.3f}")
        print(f"Adaptive (RF + Rule) - Mean: {np.mean(adaptive_rewards_rf_rule):.3f}, Std: {np.std(adaptive_rewards_rf_rule):.3f}")
        print(f"Adaptive (LR + QL) - Mean: {np.mean(adaptive_rewards_lr_ql):.3f}, Std: {np.std(adaptive_rewards_lr_ql):.3f}")

        # 绘制性能对比图
        plt.figure(figsize=(12, 6))
        window = 10
        plt.plot(np.convolve(fixed_rewards, np.ones(window)/window, mode='valid'), label='Fixed')
        plt.plot(np.convolve(threshold_rewards, np.ones(window)/window, mode='valid'), label='Threshold')
        plt.plot(np.convolve(adaptive_rewards_lstm_ql, np.ones(window)/window, mode='valid'), label='Adaptive (LSTM + QL)')
        plt.plot(np.convolve(adaptive_rewards_arima_sarsa, np.ones(window)/window, mode='valid'), label='Adaptive (ARIMA + SARSA)')
        plt.plot(np.convolve(adaptive_rewards_transformer_pg, np.ones(window)/window, mode='valid'), label='Adaptive (Transformer + PG)')
        plt.plot(np.convolve(adaptive_rewards_rf_rule, np.ones(window)/window, mode='valid'), label='Adaptive (RF + Rule)')
        plt.plot(np.convolve(adaptive_rewards_lr_ql, np.ones(window)/window, mode='valid'), label='Adaptive (LR + QL)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (10-episode window)')
        plt.title('Resource Management Methods Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误堆栈:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()