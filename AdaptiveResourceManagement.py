import csv
import numpy as np
from collections import deque, namedtuple
import random
import os # Needed for environment variable for reproducibility
from typing import List, Dict, Tuple, Deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical # For PPO/REINFORCE
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import copy # For Federated Learning target networks

# --- Function to set seeds ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass
    print(f"Set seed to {seed} for reproducibility.")

# --- End Seed Function ---

# Ignore warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="std(): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Warm-start fitting without increasing n_estimators does not fit new trees.")


# Transition tuple for Replay Buffer (used by DDPG)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- 预测器接口 ---
class IDemandPredictor:
    def predict(self, data: np.ndarray) -> float: raise NotImplementedError
    def get_model(self): return self if isinstance(self, nn.Module) else None

# --- 分配器接口 ---
class IResourceAllocator:
    def select_action(self, state) -> any: raise NotImplementedError
    def update(self, state, action, reward: float, next_state, done: bool): raise NotImplementedError
    def finish_episode(self): pass
    def get_networks(self): return None

# --- Replay Buffer (for DDPG) ---
class ReplayBuffer:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size):
        actual_batch_size = min(batch_size, len(self.memory)); return random.sample(self.memory, actual_batch_size) if actual_batch_size > 0 else []
    def __len__(self): return len(self.memory)

# --- Ornstein-Uhlenbeck Noise (for DDPG exploration) ---
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2): self.mu = mu * np.ones(size); self.theta = theta; self.sigma = sigma; self.size = size; self.reset()
    def reset(self): self.state = copy.copy(self.mu)
    def sample(self): x = self.state; noise = self.sigma * np.random.randn(self.size); dx = self.theta * (self.mu - x) + noise; self.state = x + dx; return self.state.copy()

# --- 传统资源管理器 (修正后) ---
class TraditionalResourceManager:
    def __init__(self, n_resources: int, method: str = 'fixed'):
        self.n_resources = n_resources; self.method = method; self.threshold = 0.5; self.last_raw_data_point = 2.0 # Needed for consistent reward calc if called externally
    def select_action(self, raw_data: np.ndarray, n_actions: int = 3) -> int: # Added n_actions, default 3
        """Selects an action index based on the traditional strategy."""
        self.last_raw_data_point = raw_data.item() # Store for external reward calc
        if self.method == 'fixed': action_level = self.calculate_fixed_action()
        else: action_level = self.calculate_threshold_action(raw_data)
        # Map resource level (0 to n_res-1) to action index (0 to n_actions-1)
        if self.n_resources <= 1: action_index = (n_actions - 1) // 2 # Default middle for single resource level
        elif action_level == 0: action_index = 0 # Low action
        elif action_level == self.n_resources - 1: action_index = n_actions - 1 # High action
        else: action_index = (n_actions - 1) // 2 # Mid action for intermediate levels
        return max(0, min(n_actions - 1, int(action_index))) # Ensure valid action index
    def calculate_fixed_action(self) -> int: return self.n_resources // 2 # Returns resource level
    def calculate_threshold_action(self, data: np.ndarray) -> int: allocation_level = (self.n_resources - 1) if np.mean(data) > self.threshold else 0; return int(allocation_level) # Returns resource level

# --- 1. 数据预处理模块 ---
class DataPreprocessor:
    def __init__(self, window_size: int = 10): self.window_size = window_size; self.data_buffer = deque(maxlen=window_size); self.mean = 0.0; self.std = 1.0
    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        current_val = raw_data.item(); self.data_buffer.append(current_val)
        if len(self.data_buffer) > 1: buffer_array = np.array(list(self.data_buffer)); self.mean = np.mean(buffer_array); self.std = np.std(buffer_array) + 1e-8
        elif len(self.data_buffer) == 1: self.mean = current_val; self.std = 1e-8
        else: self.mean = 0.0; self.std = 1.0
        if self.std < 1e-8: normalized_buffer = [0.0] * len(self.data_buffer)
        else: normalized_buffer = [(x - self.mean) / self.std for x in self.data_buffer]
        padded_normalized_buffer = list(normalized_buffer);
        while len(padded_normalized_buffer) < self.window_size: padded_normalized_buffer.insert(0, 0.0)
        output_window = padded_normalized_buffer[-self.window_size:]
        output_array = np.array(output_window).reshape(1, self.window_size, 1); return np.nan_to_num(output_array, nan=0.0, posinf=0.0, neginf=0.0)

# --- 2. 需求预测模块 ---
# 2.1 LSTM Predictor (Corrected Indentation)
class LSTMPredictor(nn.Module, IDemandPredictor):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None
    def init_hidden(self, batch_size=1, device='cpu'):
        # Correct indentation
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))
    def forward(self, x):
        # Correct indentation
        current_device = x.device
        batch_size = x.shape[0]
        # Correct indentation for the if statement
        if self.hidden is None or batch_size != self.hidden[0].shape[1] or self.hidden[0].device != current_device:
            # Correct indentation for the line inside the if block
            self.hidden = self.init_hidden(batch_size, device=current_device)
        # Correct indentation for the rest of the method
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        last_time_step_output = lstm_out[:, -1, :]
        predictions = self.fc(last_time_step_output)
        return predictions
    def predict(self, data: np.ndarray) -> float:
        self.eval(); device = next(self.parameters()).device
        with torch.no_grad(): data_tensor = torch.FloatTensor(data).to(device); prediction = self.forward(data_tensor).item()
        return float(prediction) if np.isfinite(prediction) else 0.0
    def get_model(self): return self

# 2.2 ARIMA 预测器
class ARIMAPredictor(IDemandPredictor):
    def __init__(self, order=(5, 1, 0), history_len=50):
        self.order = order
        self.history = deque(maxlen=history_len)

    def predict(self, data: np.ndarray) -> float:
        # --- Corrected Indentation ---
        latest_normalized_value = data[0, -1, 0]

        # Correct indentation for the if statement and its body
        if np.isfinite(latest_normalized_value):
            self.history.append(latest_normalized_value)
        # else: # Optional: handle non-finite case if needed
        #    pass

        # Correct indentation for subsequent lines
        min_hist = max(self.order[0], self.order[1] + 1, 10)
        finite_history = [float(x) for x in self.history if np.isfinite(x)]

        # Correct indentation for the outer if/else block
        if len(finite_history) >= min_hist:
            try:
                # Correct indentation for lines within try
                model = ARIMA(finite_history, order=self.order)
                model_fit = model.fit()
                output = model_fit.forecast()
                pred = output[0] if isinstance(output, (list, np.ndarray)) else output
                return float(pred) if np.isfinite(pred) else 0.0
            except Exception:
                # Correct indentation for lines within except
                # print(f"ARIMA prediction error. Falling back.") # Optional debug print
                return float(finite_history[-1]) if finite_history else 0.0
        else:
            # Correct indentation for lines within else
             return float(finite_history[-1]) if finite_history else 0.0

# 2.3 简化的 Transformer 预测器
class TransformerPredictor(nn.Module, IDemandPredictor):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int, seq_len: int):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_size, d_model)
        # Simple fixed positional encoding
        pos_encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoder', pos_encoding.unsqueeze(0))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, src):
        # Correct indentation for forward method
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = src + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.fc(output[:, -1, :])
        return output

    def predict(self, data: np.ndarray) -> float:
        # Correct indentation for predict method
        self.eval()
        device = next(self.parameters()).device
        # Correct indentation for with statement
        with torch.no_grad():
            # Correct indentation for lines inside 'with' block
            data_tensor = torch.FloatTensor(data).to(device)
            prediction = self.forward(data_tensor).item()
        # Correct indentation for return statement
        return float(prediction) if np.isfinite(prediction) else 0.0

    def get_model(self):
        # Correct indentation for get_model method
        return self

# 2.4 随机森林预测器
class RandomForestPredictor(IDemandPredictor):
    def __init__(self, n_estimators=100, window_size=10, history_len=100, seed=None):
        # Correct indentation for __init__ method body
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=seed
        )
        self.window_size = window_size
        self.X_history = deque(maxlen=history_len)
        self.y_history = deque(maxlen=history_len)
        self.fitted = False
        self.min_samples_for_fit = max(10, window_size + 1)
        self.previous_features = None

    def predict(self, data: np.ndarray) -> float:
        # Correct indentation for predict method body
        current_features = data.flatten()
        prediction = 0.0

        # Correct indentation for the 'if' statement
        if self.fitted and np.all(np.isfinite(current_features)):
             # Correct indentation for the 'try...except' block inside 'if'
             try:
                 prediction = self.model.predict(current_features.reshape(1, -1))[0]
             except Exception:
                 prediction = 0.0 # Fallback

        # Correct indentation for subsequent lines within predict method
        latest_normalized_value = data[0, -1, 0]
        if np.isfinite(latest_normalized_value) and self.previous_features is not None and np.all(np.isfinite(self.previous_features)):
            # Correct indentation inside this 'if'
            self.X_history.append(self.previous_features)
            self.y_history.append(latest_normalized_value)

        if np.all(np.isfinite(current_features)):
            # Correct indentation inside this 'if'
            self.previous_features = current_features
        else:
            # Correct indentation inside 'else'
            self.previous_features = None

        # Correct indentation for the final 'if' block
        if len(self.X_history) >= self.min_samples_for_fit and len(self.X_history) == len(self.y_history):
             # Correct indentation inside this 'if'
             X_train = np.array(list(self.X_history))
             y_train = np.array(list(self.y_history))
             if np.all(np.isfinite(X_train)) and np.all(np.isfinite(y_train)):
                 # Correct indentation for nested 'if' and 'try...except'
                 try:
                     self.model.fit(X_train, y_train)
                     self.fitted = True
                 except Exception:
                     self.fitted = False
             else:
                 self.fitted = False

        # Correct indentation for the return statement
        return float(prediction) if np.isfinite(prediction) else 0.0

# 2.5 线性回归预测器
class LinearRegressionPredictor(IDemandPredictor):
    def __init__(self, window_size=10, history_len=100):
        # Correct indentation for __init__ method body
        self.model = LinearRegression()
        self.window_size = window_size
        self.X_history = deque(maxlen=history_len)
        self.y_history = deque(maxlen=history_len)
        self.fitted = False
        self.min_samples_for_fit = max(2, window_size + 1)
        self.previous_features = None

    def predict(self, data: np.ndarray) -> float:
        # Correct indentation for predict method body
        current_features = data.flatten()
        prediction = 0.0

        # Correct indentation for the 'if' statement (line 288)
        if self.fitted and np.all(np.isfinite(current_features)):
            # Correct indentation for the 'try...except' block inside 'if'
            try:
                prediction = self.model.predict(current_features.reshape(1, -1))[0]
            except Exception:
                 prediction = 0.0

        # Correct indentation for subsequent lines within predict method
        latest_normalized_value = data[0, -1, 0]
        if np.isfinite(latest_normalized_value) and self.previous_features is not None and np.all(np.isfinite(self.previous_features)):
            # Correct indentation inside this 'if'
            self.X_history.append(self.previous_features)
            self.y_history.append(latest_normalized_value)

        # Correct indentation for the next 'if/else' block
        if np.all(np.isfinite(current_features)):
            self.previous_features = current_features
        else:
            self.previous_features = None

        # Correct indentation for the final 'if' block
        if len(self.X_history) >= self.min_samples_for_fit and len(self.X_history) == len(self.y_history):
            # Correct indentation inside this 'if'
            X_train = np.array(list(self.X_history))
            y_train = np.array(list(self.y_history))
            if np.all(np.isfinite(X_train)) and np.all(np.isfinite(y_train)):
                 # Correct indentation for nested 'if' and 'try...except'
                 try:
                     self.model.fit(X_train, y_train)
                     self.fitted = True
                 except Exception:
                     self.fitted = False
            else:
                 self.fitted = False

        # Correct indentation for the return statement
        return float(prediction) if np.isfinite(prediction) else 0.0

# --- 3. 资源分配模块 ---
# 3.1 Q-Learning 分配器
class QLearningAllocator(IResourceAllocator):
    def __init__(self, state_dim: int, n_actions: int, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        # Correct indentation for __init__ body
        self.n_states = state_dim
        self.n_actions = n_actions
        self.q_table = np.zeros((self.n_states, n_actions))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

    def select_action(self, state: int) -> int:
        # Correct indentation for select_action body
        state = max(0, min(self.n_states - 1, int(state)))
        q_row = self.q_table[state]

        # Correct indentation for the 'if' statement (line 345)
        if random.random() < self.epsilon or np.isnan(q_row).any():
            # Correct indentation for the line inside the 'if' block
            return random.randint(0, self.n_actions - 1)

        # Correct indentation for subsequent lines within select_action
        best_actions = np.flatnonzero(q_row == q_row.max())
        return random.choice(best_actions)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Correct indentation for update body
        if not np.isfinite(reward):
            return # Keep indented under the method def
        state = max(0, min(self.n_states - 1, int(state)))
        next_state = max(0, min(self.n_states - 1, int(next_state)))
        action = max(0, min(self.n_actions - 1, int(action)))
        old_value = self.q_table[state, action]
        next_q_row = self.q_table[next_state]
        next_max = 0
        # Correct indentation for the 'if'/'elif' block
        if not done and not np.isnan(next_q_row).any():
             next_max = np.max(next_q_row)
        # Optional: Handle NaN in next state Q-values explicitly if desired
        # elif not done:
        #      next_max = 0 # Default if next state has NaN

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.gamma * next_max)
        # Correct indentation for the final 'if'
        if np.isfinite(new_value):
            self.q_table[state, action] = new_value
        # else: print(f"Warning: Calculated NaN Q-value update for ({state}, {action}). Skipping.")

    def finish_episode(self):
        # Correct indentation for finish_episode body
        self.epsilon = max(0.01, self.epsilon * 0.995)

# 3.2 SARSA 分配器
class SARSAAllocator(IResourceAllocator):
    def __init__(self, state_dim: int, n_actions: int, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        # Correct indentation for __init__ body
        self.n_states = state_dim
        self.n_actions = n_actions
        self.q_table = np.zeros((self.n_states, n_actions))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

    def select_action(self, state: int) -> int:
        # Correct indentation for select_action body
        state = max(0, min(self.n_states - 1, int(state)))
        q_row = self.q_table[state]

        # Correct indentation for the 'if' statement (line 398)
        if random.random() < self.epsilon or np.isnan(q_row).any():
            # Correct indentation for the line inside the 'if' block
            return random.randint(0, self.n_actions - 1)

        # Correct indentation for subsequent lines within select_action
        best_actions = np.flatnonzero(q_row == q_row.max())
        return random.choice(best_actions)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Correct indentation for update body
        if not np.isfinite(reward):
            return # Keep indented under the method def

        state = max(0, min(self.n_states - 1, int(state)))
        action = max(0, min(self.n_actions - 1, int(action)))
        next_state = max(0, min(self.n_states - 1, int(next_state)))

        next_action = self.select_action(next_state) # On-policy selection of a'

        old_value = self.q_table[state, action]
        next_q_value = 0
        # Correct indentation for the 'if'/'elif' block
        # Check index validity before accessing q_table
        if not done and next_state < self.n_states and next_action < self.n_actions and not np.isnan(self.q_table[next_state, next_action]):
            next_q_value = self.q_table[next_state, next_action]
        # Optional: Handle NaN case explicitly if needed
        # elif not done:
        #      next_q_value = 0

        new_value = old_value + self.learning_rate * (reward + self.gamma * next_q_value - old_value)

        # Correct indentation for the final 'if'
        if np.isfinite(new_value):
            self.q_table[state, action] = new_value
        # else: print(f"Warning: Calculated NaN SARSA Q-value update for ({state}, {action}). Skipping.")

    def finish_episode(self):
        # Correct indentation for finish_episode body
        self.epsilon = max(0.01, self.epsilon * 0.995)

# 3.3 策略梯度分配器 (REINFORCE)
class PolicyGradientAllocator(IResourceAllocator):
    def __init__(self, state_dim: int, n_actions: int, learning_rate=0.01, gamma=0.99):
        # Correct indentation for __init__ body
        self.input_dim = 1
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(),
            nn.Linear(64, n_actions), nn.Softmax(dim=-1)
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []

    def select_action(self, state: int) -> int:
        # Correct indentation for select_action body
        state_tensor = torch.tensor([[float(state)]], dtype=torch.float32).to(self.device)
        self.policy_network.eval()
        action = random.randint(0, self.n_actions - 1) # Default
        log_prob = torch.tensor(0.0) # Default

        # Correct indentation for the 'try...except' block
        try:
            # Correct indentation for lines inside 'try'
            with torch.no_grad():
                 # Correct indentation for lines inside 'with'
                 probs = self.policy_network(state_tensor)
                 if torch.isnan(probs).any() or torch.isinf(probs).any() or not torch.all(probs >= 0):
                     return action # Return default random action
                 action_dist = Categorical(probs)
                 action_tensor = action_dist.sample()
                 log_prob = action_dist.log_prob(action_tensor)
                 action = action_tensor.item()
        except Exception as e:
            # Correct indentation for lines inside 'except'
            print(f"Error during PG select_action: {e}. Returning random action.")
            return random.randint(0, self.n_actions - 1)

        # Correct indentation for subsequent lines
        self.policy_network.train()
        self.log_probs.append(log_prob.cpu()) # Store log_prob on CPU
        return action

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Correct indentation for update body
        if np.isfinite(reward):
            self.rewards.append(reward)
        else:
            # Ensure log_probs is not empty before popping
            if self.log_probs:
                self.log_probs.pop()

    def finish_episode(self):
        # Correct indentation for finish_episode body
        if not self.rewards or not self.log_probs or len(self.rewards) != len(self.log_probs):
             self.rewards = []
             self.log_probs = []
             return

        R = 0
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)

        returns_tensor = torch.tensor(list(returns), dtype=torch.float32)
        if returns_tensor.numel() == 0:
             self.rewards = []
             self.log_probs = []
             return

        if returns_tensor.numel() > 1:
             mean = returns_tensor.mean()
             std = returns_tensor.std()
             if std > 1e-8:
                 returns_tensor = (returns_tensor - mean) / std
             else:
                 returns_tensor = returns_tensor - mean

        # Concatenate log_probs which should be a list of 0-dim or 1-dim tensors
        try:
            # Ensure tensors are moved to the correct device if necessary (already on CPU)
            valid_log_probs = torch.cat([lp.view(1) for lp in self.log_probs]) # Ensure they are at least 1D before cat
        except Exception as e:
            print(f"Error concatenating log_probs in PG: {e}")
            self.rewards = []; self.log_probs = []; return

        valid_returns = returns_tensor

        # Check alignment after potential reward/log_prob mismatch handling
        num_steps = min(len(valid_log_probs), len(valid_returns))
        if num_steps == 0:
            self.rewards = []; self.log_probs = []; return
        valid_log_probs = valid_log_probs[:num_steps]
        valid_returns = valid_returns[:num_steps]


        if torch.isnan(valid_log_probs).any() or torch.isnan(valid_returns).any():
            self.rewards = []
            self.log_probs = []
            return

        policy_loss = (-valid_log_probs * valid_returns).sum()

        if torch.isnan(policy_loss):
             self.rewards = []
             self.log_probs = []
             return

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

# 3.4 基于规则的分配器
class RuleBasedAllocator(IResourceAllocator):
    def __init__(self, n_resources: int, n_actions: int, threshold: float = 0.0):
        # Correct indentation for __init__ body
        self.n_resources = n_resources
        self.n_actions = n_actions
        self.threshold = threshold

    def select_action(self, predicted_demand: float) -> int:
        # Correct indentation for select_action body
        action = (self.n_actions - 1) // 2 # Default middle action

        # Correct indentation for the 'if/elif' block (line 575 is the 'if')
        if predicted_demand > self.threshold + 0.5:
            # Correct indentation for line inside 'if'
            action = self.n_actions - 1
        elif predicted_demand < self.threshold - 0.5:
            # Correct indentation for line inside 'elif'
            action = 0

        # Correct indentation for the return statement
        return max(0, min(self.n_actions - 1, int(action)))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Correct indentation for update body (even if it's just 'pass')
        pass

    def finish_episode(self):
        # Correct indentation for finish_episode body (even if it's just 'pass')
        pass

# 3.5 PPO 分配器 (修正后)
class PPOAllocator(IResourceAllocator):
    def __init__(self, state_dim: int, n_actions: int, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2, gae_lambda=0.95):
        # ... (init code - already checked) ...
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions
        self.input_dim = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO using device: {self.device}")
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(), 
            nn.Linear(64, n_actions), nn.Softmax(dim=-1)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(), 
            nn.Linear(64, 1)
        ).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.MseLoss = nn.MSELoss()
        self.memory = []
        self.current_log_prob = None
        self.current_value = None

    def store_transition(self, state, action, reward, log_prob, value, done):
        # ... (store_transition code - already checked) ...
        if not all(np.isfinite([state, action, reward, log_prob.item(), value.item()])):
            return
        self.memory.append((float(state), int(action), float(reward), log_prob.item(), value.item(), bool(done)))

    def select_action(self, state: int) -> int:
        # ... (select_action code - already checked) ...
        self.actor.eval()
        self.critic.eval()
        action = random.randint(0, self.n_actions - 1)
        log_prob_for_storage = torch.tensor(-1e6)
        value_for_storage = torch.tensor(0.0)
        try:
            with torch.no_grad():
                state_tensor = torch.tensor([[float(state)]], dtype=torch.float32).to(self.device)
                action_probs = self.actor(state_tensor)
                value = self.critic(state_tensor)
            is_invalid_probs = torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or not torch.all(action_probs >= 0) or abs(action_probs.sum() - 1.0) > 1e-5
            if not is_invalid_probs:
                dist = Categorical(probs=action_probs)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
                action = action_tensor.item()
                log_prob_for_storage = log_prob.detach().cpu()
                value_for_storage = value.detach().squeeze().cpu()
        except Exception:
            pass
        self.actor.train()
        self.critic.train()
        self.current_log_prob = log_prob_for_storage
        self.current_value = value_for_storage
        return action

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # ... (update code - already checked) ...
        log_prob = self.current_log_prob if self.current_log_prob is not None else torch.tensor(-1e6)
        value = self.current_value if self.current_value is not None else torch.tensor(0.0)
        self.store_transition(state, action, reward, log_prob, value, done)
        self.current_log_prob = None
        self.current_value = None

    def compute_gae_and_returns(self, rewards, values, dones):
        # Correct indentation for compute_gae_and_returns body
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0
        if not dones[-1].item() and values.numel() > 0:
             next_value = values[-1].item()

        next_value_tensor = torch.tensor([next_value], dtype=torch.float32, device=values.device)
        if values.numel() > 0:
            values_with_bootstrap = torch.cat((values, next_value_tensor), dim=0)
        else:
            values_with_bootstrap = next_value_tensor

        # Correct indentation for the 'for' loop
        for i in reversed(range(len(rewards))):
            # Correct indentation for lines inside the loop
            reward = rewards[i].item()
            done_mask = 1.0 - dones[i].item()
            value = 0.0
            next_val = 0.0
            # Correct indentation for the 'if' statement (line 635)
            if i < values_with_bootstrap.shape[0]:
                # Correct indentation for the line inside 'if'
                value = values_with_bootstrap[i].item()
            # Correct indentation for the next 'if' statement
            if i + 1 < values_with_bootstrap.shape[0]:
                # Correct indentation for the line inside 'if'
                next_val = values_with_bootstrap[i+1].item()

            # Correct indentation for subsequent calculations
            delta = reward + self.gamma * next_val * done_mask - value
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + value)

        # Correct indentation for the final tensor conversions and return
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=values.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=values.device)
        return advantages_tensor, returns_tensor

    def finish_episode(self):
        # ... (finish_episode code - already checked) ...
        if not self.memory:
            return
        try:
            states_list, actions_list, rewards_list, old_log_probs_list, old_values_list, dones_list = zip(*self.memory)
        except ValueError:
            self.memory = []
            return
        try:
            states = torch.tensor(states_list, dtype=torch.float32).view(-1, 1).to(self.device)
            actions = torch.tensor(actions_list, dtype=torch.long).view(-1).to(self.device)
            rewards = torch.tensor(rewards_list, dtype=torch.float32).view(-1).to(self.device)
            old_log_probs = torch.tensor(old_log_probs_list, dtype=torch.float32).view(-1).to(self.device)
            old_values = torch.tensor(old_values_list, dtype=torch.float32).view(-1).to(self.device)
            dones = torch.tensor(dones_list, dtype=torch.float32).view(-1).to(self.device)
        except Exception:
            self.memory = []
            return
        if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(rewards).any() or torch.isnan(old_log_probs).any() or torch.isnan(old_values).any() or torch.isnan(dones).any():
            self.memory = []
            return
        try:
            advantages, returns = self.compute_gae_and_returns(rewards, old_values, dones)
        except Exception:
            self.memory = []
            return
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            self.memory = []
            return
        if advantages.numel() > 1:
            std_adv = advantages.std()
            if std_adv > 1e-8:
                advantages = (advantages - advantages.mean()) / (std_adv + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        if torch.isnan(advantages).any():
            self.memory = []
            return
        self.actor.train()
        self.critic.train()
        for k_epoch in range(self.K_epochs):
            try:
                action_probs = self.actor(states)
                is_invalid_probs = torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or not torch.all(action_probs >= 0) or abs(action_probs.sum(dim=-1).mean() - 1.0) > 1e-5
                if is_invalid_probs:
                    continue
                dist = Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                dist_entropy = dist.entropy().mean()
                values = self.critic(states).squeeze(-1)
                if torch.isnan(log_probs).any() or torch.isnan(values).any() or torch.isnan(dist_entropy):
                    continue
                ratios = torch.exp(log_probs - old_log_probs)
                if torch.isnan(ratios).any() or torch.isinf(ratios).any():
                    continue
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(values, returns)
                if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isnan(dist_entropy):
                    continue
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
            except ValueError:
                break
            except Exception:
                break
        self.memory = []

# 3.6 DDPGAllocator
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGActor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, s):
        x = F.relu(self.layer_1(s))
        x = F.relu(self.layer_2(x))
        return torch.sigmoid(self.layer_3(x)) * self.max_action

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)
    
    def forward(self, s, a):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if a.dim() == 0:
            a = a.unsqueeze(0).unsqueeze(0)
        sa = torch.cat([s, a], 1)
        x = F.relu(self.layer_1(sa))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class DDPGAllocator(IResourceAllocator):
    def __init__(self, state_dim: int, n_actions: int, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, buffer_size=10000, batch_size=64, noise_sigma=0.1):
        self.state_dim = 1
        self.action_dim = 1
        self.n_discrete_actions = n_actions
        self.max_action = float(n_actions - 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DDPG using device: {self.device}")
        self.actor = DDPGActor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = DDPGCritic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(self.action_dim, sigma=noise_sigma * self.max_action)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
    
    def select_action(self, state: int) -> float:
        self.actor.eval()
        action = 0.0
        try:
            with torch.no_grad():
                state_tensor = torch.tensor([[float(state)]], dtype=torch.float32).to(self.device)
                action = self.actor(state_tensor).cpu().item()
        except Exception:
            pass
        self.actor.train()
        noise = self.noise.sample()[0]
        action = np.clip(action + noise, 0, self.max_action)
        return float(action)
    
    def update(self, state: int, action: float, reward: float, next_state: int, done: bool):
        if not all(np.isfinite([state, action, reward, next_state])):
            return
        self.replay_buffer.push(
            torch.tensor([[float(state)]], dtype=torch.float32),
            torch.tensor([[action]], dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([[float(next_state)]], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool)
        )
        if len(self.replay_buffer) >= self.batch_size:
            self.learn()
    
    def learn(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        if not transitions:
            return
        batch = Transition(*zip(*transitions))
        try:
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward).to(self.device)
            next_state_batch = torch.cat(batch.next_state).to(self.device)
            done_batch = torch.cat(batch.done).float().to(self.device)
        except Exception:
            return
        
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
        
        with torch.no_grad():
            next_action_batch = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_action_batch)
            target_Q = reward_batch.unsqueeze(1) + ((1.0 - done_batch.unsqueeze(1)) * self.gamma * target_Q)
            target_Q = torch.nan_to_num(target_Q)
        
        current_Q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        if torch.isnan(critic_loss):
            return
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        for params in self.critic.parameters():
            params.requires_grad = False
        
        actor_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actor_actions).mean()
        
        # --- Corrected NaN check block ---
        if torch.isnan(actor_loss):
            for params in self.critic.parameters():
                params.requires_grad = True # Unfreeze critic FIRST
            return # THEN return
        # --- End Correction ---
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        for params in self.critic.parameters():
            params.requires_grad = True # Unfreeze critic AFTER update
        
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def finish_episode(self):
        self.noise.reset()

# --- 4. 自适应控制模块 ---
class AdaptiveController:
    def __init__(self, history_len=20):
        self.performance_history = deque(maxlen=history_len)
        self.adaptation_threshold = 0.6
        self.min_performance_len = 10
    
    def evaluate_performance(self, metrics: Dict) -> float:
        performance = metrics.get('reward', 0.0)
        self.performance_history.append(performance)
        return performance
    
    def need_adaptation(self) -> bool:
        if len(self.performance_history) < self.min_performance_len:
            return False
        return np.mean(list(self.performance_history)) < self.adaptation_threshold

# --- 5. 主框架 (使用修改后的奖励函数) ---
class AdaptiveResourceManager:
    def __init__(self, n_resources: int, n_actions: int, predictor: IDemandPredictor, allocator: IResourceAllocator, adaptation_enabled=True):
        self.n_resources = n_resources
        self.n_actions = n_actions
        self.preprocessor = DataPreprocessor(window_size=10)
        self.predictor = predictor
        self.allocator = allocator
        self.controller = AdaptiveController()
        self.current_state = self.n_resources // 2
        self.adaptation_enabled = adaptation_enabled
        self.last_raw_data_point = 2.0
        self.threshold_for_reward = 0.5 # Threshold on raw data for reward
    
    def step(self, raw_data: np.ndarray) -> Tuple[int, float, int]: # Modified to return action
        self.last_raw_data_point = raw_data.item()
        processed_data = self.preprocessor.preprocess(raw_data)
        predicted_demand = 0.0
        try:
            predicted_demand = self.predictor.predict(processed_data)
            predicted_demand = float(np.nan_to_num(predicted_demand, nan=0.0, posinf=1.0, neginf=-1.0))
        except Exception:
            predicted_demand = 0.0
        
        state_for_allocator = int(self.current_state)
        action = random.randint(0, self.n_actions - 1)
        continuous_action_taken = None
        
        try:
            if isinstance(self.allocator, RuleBasedAllocator):
                action = self.allocator.select_action(predicted_demand)
            elif isinstance(self.allocator, DDPGAllocator):
                continuous_action = self.allocator.select_action(state_for_allocator)
                continuous_action_taken = continuous_action
                action = int(np.round(continuous_action))
            else:
                action = self.allocator.select_action(state_for_allocator)
        except Exception:
            action = random.randint(0, self.n_actions - 1)
        
        action = max(0, min(self.n_actions - 1, int(action))) # Final discrete action
        reward = self.simulate_environment_feedback(action)
        
        mid_action_floor = (self.n_actions - 1) // 2
        mid_action_ceil = (self.n_actions - 1 + 1) // 2
        state_change = 0
        
        if action < mid_action_floor:
            state_change = -1
        elif action >= mid_action_ceil:
            state_change = 1
        
        next_state = max(0, min(self.n_resources - 1, self.current_state + state_change))
        done = False
        
        update_action = continuous_action_taken if isinstance(self.allocator, DDPGAllocator) and continuous_action_taken is not None else action
        try:
            if np.isfinite(reward):
                self.allocator.update(self.current_state, update_action, reward, next_state, done)
        except Exception:
            pass
        
        if self.adaptation_enabled:
            metrics = {'reward': reward}
            self.controller.evaluate_performance(metrics)
            if self.controller.need_adaptation():
                self.adapt_system()
        
        self.current_state = next_state
        return self.current_state, reward, action # Return chosen action
    
    def simulate_environment_feedback(self, discrete_action: int) -> float:
        actual_demand_is_high = self.last_raw_data_point > self.threshold_for_reward
        ideal_target_action = (self.n_actions - 1) // 2
        
        if actual_demand_is_high:
            ideal_target_action = self.n_actions - 1
        else:
            ideal_target_action = 0
        
        k = 0.7
        reward = np.exp(-k * abs(discrete_action - ideal_target_action))
        return float(reward)
    
    def adapt_system(self):
        adapted = False
        alloc_type = type(self.allocator).__name__
        
        if hasattr(self.allocator, 'epsilon'):
            new_epsilon = max(0.01, self.allocator.epsilon * 0.9)
            if new_epsilon < self.allocator.epsilon:
                self.allocator.epsilon = new_epsilon
                adapted = True
        elif hasattr(self.allocator, 'noise') and hasattr(self.allocator.noise, 'sigma'):
            current_sigma = self.allocator.noise.sigma
            min_sigma = 0.01 * getattr(self.allocator, 'max_action', 1.0)
            new_sigma = max(min_sigma, current_sigma * 0.95)
            if new_sigma < current_sigma:
                self.allocator.noise.sigma = new_sigma
                adapted = True
        
        if adapted:
            self.controller.performance_history.clear()

# --- Experiment Runner (Modified for consistent reward calculation) ---
def run_experiment(run_id, n_resources = 7, n_actions = 3, episodes = 500, seed=None):
    current_seed = seed if seed is not None else run_id
    set_seed(current_seed)
    print(f"\n--- Starting Run {run_id} (Seed: {current_seed}) ---")
    print(f"Resource Levels (States): {n_resources}, Discrete Actions: {n_actions}, Episodes: {episodes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for NN models.")

    # Initialize managers
    fixed_manager = TraditionalResourceManager(n_resources=n_resources, method='fixed')
    threshold_manager = TraditionalResourceManager(n_resources=n_resources, method='threshold')

    # Initialize predictors and adaptive managers
    def create_lstm():
        return LSTMPredictor(input_size=1, hidden_size=32, output_size=1).to(device)
    
    def create_transformer():
        return TransformerPredictor(input_size=1, d_model=16, nhead=2, num_layers=1, output_size=1, seq_len=10).to(device)
    
    def create_rf():
        return RandomForestPredictor(window_size=10, seed=current_seed)
    
    adaptive_managers = { # Keep adaptive managers separate for calling step/finish_episode
        "Adaptive (LSTM + QL)": AdaptiveResourceManager(n_resources, n_actions, create_lstm(), QLearningAllocator(state_dim=n_resources, n_actions=n_actions)),
        "Adaptive (ARIMA + SARSA)": AdaptiveResourceManager(n_resources, n_actions, ARIMAPredictor(), SARSAAllocator(state_dim=n_resources, n_actions=n_actions)),
        "Adaptive (Trans + PG)": AdaptiveResourceManager(n_resources, n_actions, create_transformer(), PolicyGradientAllocator(state_dim=n_resources, n_actions=n_actions)),
        "Adaptive (RF + Rule)": AdaptiveResourceManager(n_resources, n_actions, create_rf(), RuleBasedAllocator(n_resources=n_resources, n_actions=n_actions)),
        "Adaptive (LR + QL)": AdaptiveResourceManager(n_resources, n_actions, LinearRegressionPredictor(window_size=10), QLearningAllocator(state_dim=n_resources, n_actions=n_actions)),
        "Adaptive (LSTM + PPO)": AdaptiveResourceManager(n_resources, n_actions, create_lstm(), PPOAllocator(state_dim=n_resources, n_actions=n_actions)),
        "Adaptive (LSTM + DDPG)": AdaptiveResourceManager(n_resources, n_actions, create_lstm(), DDPGAllocator(state_dim=n_resources, n_actions=n_actions)),
    }
    all_manager_names = ["Fixed", "Threshold"] + list(adaptive_managers.keys())

    # Logging
    rewards_log = {name: [] for name in all_manager_names}
    episode_data_log = []
    last_raw_data_point_for_reward = 2.0 # Initialize for first step reward calc
    reward_threshold = 0.5 # Centralized threshold for reward calculation

    # Simulation loop
    for episode in range(episodes):
        time_step = episode
        raw_data_point = 0.5 * np.sin(time_step / 50.0) + np.random.randn() * 0.2 + 2.0
        raw_data = np.array([raw_data_point])

        # --- Determine Ideal Action based on PREVIOUS step's actual data ---
        actual_demand_is_high = last_raw_data_point_for_reward > reward_threshold
        ideal_target_action = (n_actions - 1) // 2
        if actual_demand_is_high:
            ideal_target_action = n_actions - 1
        else:
            ideal_target_action = 0
        # --- ---

        # --- Get actions and run step for all managers ---
        actions_taken = {}
        for name in all_manager_names:
            action = 0 # Default
            manager = None
            if name == "Fixed":
                manager = fixed_manager
            elif name == "Threshold":
                manager = threshold_manager
            else:
                manager = adaptive_managers.get(name)

            try:
                if isinstance(manager, TraditionalResourceManager):
                    # Pass n_actions here if needed by select_action's mapping logic
                    actions_taken[name] = manager.select_action(raw_data, n_actions=n_actions)
                elif isinstance(manager, AdaptiveResourceManager):
                    # step now returns (next_state, reward, action)
                    _, _, action_taken_by_adaptive = manager.step(raw_data)
                    actions_taken[name] = action_taken_by_adaptive # Store the action taken
            except Exception as e:
                print(f"Error during action selection/step for {name}: {e}. Using default action 0.")
                actions_taken[name] = 0
        # --- ---

        # --- Calculate reward for ALL managers based on ideal action and action taken ---
        k_reward = 0.7
        for name, action in actions_taken.items():
            reward = np.exp(-k_reward * abs(action - ideal_target_action))
            rewards_log[name].append(float(reward) if np.isfinite(reward) else 0.0)
        # --- ---

        # Update last raw data point for the *next* step's reward calculation
        last_raw_data_point_for_reward = raw_data.item()

        # Finish episode for adaptive RL agents
        for manager in adaptive_managers.values(): # Only loop through adaptive managers
             try:
                 manager.allocator.finish_episode()
             except Exception:
                 pass

        # Log summary periodically
        if (episode + 1) % 10 == 0:
            episode_summary = {'Run ID': run_id, 'Episode': episode + 1}
            for name in all_manager_names:
                last_10 = rewards_log[name][-10:]
                episode_summary[f'{name} Reward'] = np.mean(last_10) if last_10 else 0.0
            episode_data_log.append(episode_summary)

        # Print progress
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
             print(f"  Run {run_id}, Episode {episode + 1}/{episodes} completed.")
             for name in all_manager_names:
                 rewards_to_avg = rewards_log[name][-100:]
                 avg_rew = np.mean(rewards_to_avg) if rewards_to_avg else 0.0
                 print(f"    {name}: Avg Reward (last {len(rewards_to_avg)}) = {avg_rew:.4f}")

    # Aggregate results
    overall_data = {'Run ID': run_id}
    for name, rewards in rewards_log.items():
        overall_data[f'{name} Mean'] = np.mean(rewards) if rewards else 0.0
        overall_data[f'{name} Std'] = np.std(rewards) if rewards else 0.0
    print(f"--- Run {run_id} Finished ---")
    for name in all_manager_names:
        print(f"  {name}: Overall Mean Reward = {overall_data[f'{name} Mean']:.4f}, Std Dev = {overall_data[f'{name} Std']:.4f}")
    return episode_data_log, overall_data, rewards_log

# --- Federated Learning Setup --- (Remains the same)
class FLClient:
    def __init__(self, client_id, predictor_model: LSTMPredictor, local_data_size=100, window_size=10):
        self.client_id = client_id
        self.predictor = copy.deepcopy(predictor_model).cpu()
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_bias = client_id * 0.5 - 1.0
        client_noise_std = 0.1 + client_id * 0.05
        raw_time_series = 0.5 * np.sin(np.arange(local_data_size) / 20.0) + client_bias + np.random.randn(local_data_size) * client_noise_std
        self.local_data_raw = raw_time_series.reshape(-1, 1)
        self.local_sequences = []
        self.local_targets = []
        temp_buffer = deque(maxlen=window_size)
        local_mean = np.mean(self.local_data_raw)
        local_std = np.std(self.local_data_raw) + 1e-8
        normalized_data = (self.local_data_raw - local_mean) / local_std
        for i in range(len(normalized_data)):
            temp_buffer.append(normalized_data[i, 0])
            if len(temp_buffer) == window_size and i + 1 < len(normalized_data):
                seq = np.array(list(temp_buffer)).reshape(1, window_size, 1)
                target = normalized_data[i+1].reshape(1, 1)
                if np.all(np.isfinite(seq)) and np.all(np.isfinite(target)):
                    self.local_sequences.append(torch.FloatTensor(seq))
                    self.local_targets.append(torch.FloatTensor(target))
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def set_model_weights(self, global_weights):
        self.predictor.load_state_dict(global_weights)
    
    def get_model_weights(self):
        return self.predictor.cpu().state_dict()
    
    def local_train(self, epochs=1):
        if not self.local_sequences:
            return self.get_model_weights(), 0
        self.predictor.to(self.device)
        self.predictor.train()
        total_loss = 0
        steps = 0
        for epoch in range(epochs):
            self.predictor.hidden = self.predictor.init_hidden(batch_size=1, device=self.device)
            for seq, target in zip(self.local_sequences, self.local_targets):
                seq, target = seq.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.predictor(seq)
                if torch.isnan(output).any():
                    continue
                loss = self.criterion(output, target)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                steps += 1
        avg_epoch_loss = total_loss / steps if steps > 0 else 0
        self.predictor.cpu()
        return self.get_model_weights(), avg_epoch_loss

class FLServer:
    def __init__(self, global_model: LSTMPredictor):
        self.global_model = global_model.cpu()
    
    def aggregate_weights(self, client_weights: List[Dict]):
        if not client_weights:
            return
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            try:
                valid_weights = [cw[k].float() for cw in client_weights if k in cw and isinstance(cw[k], torch.Tensor) and not torch.isnan(cw[k]).any()]
                if valid_weights:
                    stacked_weights = torch.stack(valid_weights, 0)
                    global_dict[k] = stacked_weights.mean(0)
            except KeyError:
                print(f"Warning: Key {k} not found in some client weights during aggregation. Skipping key.")
                continue
            except Exception as e:
                print(f"Error aggregating key {k}: {e}")
                continue
        try:
            self.global_model.load_state_dict(global_dict)
        except Exception as e:
            print(f"Error loading aggregated state dict: {e}")
    
    def get_global_weights(self):
        return self.global_model.state_dict()

def run_federated_experiment(num_clients=5, fl_rounds=20, local_epochs=3, seed=None):
    fl_seed = seed if seed is not None else 42
    set_seed(fl_seed)
    print(f"\n--- Starting Federated Learning Experiment (Seed: {fl_seed}) ---")
    global_lstm_predictor = LSTMPredictor(input_size=1, hidden_size=32, output_size=1).cpu()
    server = FLServer(global_lstm_predictor)
    clients = [FLClient(client_id=i, predictor_model=global_lstm_predictor, window_size=10) for i in range(num_clients)]
    fl_losses = []
    print(f"Starting {fl_rounds} FL rounds with {num_clients} clients...")
    for round_num in range(fl_rounds):
        global_weights = server.get_global_weights()
        client_weights_list = []
        round_client_losses = []
        for client in clients:
            client.set_model_weights(global_weights)
            local_weights, avg_loss = client.local_train(epochs=local_epochs)
            is_nan_weight = False
            for param in local_weights.values():
                if torch.isnan(param).any():
                    is_nan_weight = True
                    break
            if not is_nan_weight:
                client_weights_list.append(local_weights)
                round_client_losses.append(avg_loss)
        server.aggregate_weights(client_weights_list)
        avg_round_loss = np.mean(round_client_losses) if round_client_losses else 0
        fl_losses.append(avg_round_loss)
        print(f"  FL Round {round_num + 1}/{fl_rounds} - Average Client Loss: {avg_round_loss:.4f}")
    print("--- Federated Learning Experiment Finished ---")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, fl_rounds + 1), fl_losses)
    plt.xlabel("FL Round")
    plt.ylabel("Average Client Training Loss")
    plt.title(f"Federated Learning Training Loss (Seed: {fl_seed})")
    plt.grid(True)
    plt.savefig("fl_training_loss.png")
    print("Federated Learning loss plot saved as fl_training_loss.png")
    plt.close()
    return server.global_model

# --- Main Execution ---
def main():
    master_seed = 42
    num_runs = 5
    num_episodes = 1000 # Shorter episodes
    n_resources = 7
    n_actions = 3
    all_episode_data = []
    all_overall_data = []
    last_run_rewards = {}
    
    for run in range(num_runs):
        run_seed = master_seed + run
        episode_data, overall_data, rewards_log = run_experiment(run + 1, n_resources=n_resources, n_actions=n_actions, episodes=num_episodes, seed=run_seed)
        all_episode_data.extend(episode_data)
        all_overall_data.append(overall_data)
        if run == num_runs - 1:
            last_run_rewards = rewards_log
    
    base_filename = f"seed_{master_seed}_reward_mod_v3" # Updated filename indicator
    episode_csv_file = f'experiment_episodes_{base_filename}.csv'
    if all_episode_data:
        episode_fieldnames = list(all_episode_data[0].keys())
        if 'Episode' in episode_fieldnames:
            episode_fieldnames.insert(0, episode_fieldnames.pop(episode_fieldnames.index('Episode')))
        if 'Run ID' in episode_fieldnames:
            episode_fieldnames.insert(0, episode_fieldnames.pop(episode_fieldnames.index('Run ID')))
        try:
            with open(episode_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=episode_fieldnames)
                writer.writeheader()
                writer.writerows(all_episode_data)
            print(f"\nEpisodic data saved to {episode_csv_file}")
        except Exception as e:
            print(f"Error saving episodic data: {e}")
    
    overall_csv_file = f'experiment_overall_{base_filename}.csv'
    if all_overall_data:
        overall_fieldnames = list(all_overall_data[0].keys())
        if 'Run ID' in overall_fieldnames:
            overall_fieldnames.insert(0, overall_fieldnames.pop(overall_fieldnames.index('Run ID')))
        try:
            with open(overall_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=overall_fieldnames)
                writer.writeheader()
                writer.writerows(all_overall_data)
            print(f"Overall data saved to {overall_csv_file}")
        except Exception as e:
            print(f"Error saving overall data: {e}")
    
    if last_run_rewards:
        plt.figure(figsize=(15, 8))
        smoothing_window = max(1, min(50, num_episodes // 10))
        episodes_x = np.arange(num_episodes)
        for name, rewards in last_run_rewards.items():
            if rewards and len(rewards) >= smoothing_window:
                smoothed_rewards = np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
                episodes_smoothed_x = episodes_x[smoothing_window - 1:]
                plt.plot(episodes_smoothed_x, smoothed_rewards, label=name, alpha=0.8)
            elif rewards:
                plt.plot(episodes_x[:len(rewards)], rewards, label=f"{name} (Raw)", alpha=0.5, linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (Smoothed over {smoothing_window} episodes)')
        plt.title(f'Resource Management Comparison - Run {num_runs} (Seed: {run_seed}, Reward Mod V3)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f'rewards_comparison_plot_{base_filename}.png')
        print(f"\nComparison plot saved as rewards_comparison_plot_{base_filename}.png")
        plt.close()
    
    try:
        trained_fl_model = run_federated_experiment(num_clients=5, fl_rounds=20, local_epochs=3, seed=master_seed)
    except Exception as e:
        print(f"Federated Learning experiment failed: {e}")
        trained_fl_model = None
    
    if trained_fl_model:
        print("\n--- Running Experiment with FL-Trained Predictor (Reward Mod V3) ---")
        exp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fl_lstm_predictor = trained_fl_model.to(exp_device)
        fl_run_seed = master_seed + num_runs
        set_seed(fl_run_seed)
        print(f"Running FL Manager test with Seed: {fl_run_seed}")
        fl_allocator = PPOAllocator(state_dim=n_resources, n_actions=n_actions)
        fl_manager = AdaptiveResourceManager(n_resources, n_actions, fl_lstm_predictor, fl_allocator)
        fl_rewards = []
        print(f"Simulating FL Manager for {num_episodes} episodes...")
        np.random.seed(fl_run_seed)
        for episode in range(num_episodes):
            time_step = episode
            raw_data_point = 0.5 * np.sin(time_step / 50.0) + np.random.randn() * 0.2 + 2.0
            raw_data = np.array([raw_data_point])
            # Call step and get the action for logging, but use internal reward for allocator update
            _, _, action_taken = fl_manager.step(raw_data) # Need the action for centralized reward calc
            # Calculate reward externally using the action taken by the adaptive manager
            actual_demand_is_high = fl_manager.last_raw_data_point > fl_manager.threshold_for_reward # Use manager's internal state
            ideal_target_action = (n_actions - 1) // 2
            if actual_demand_is_high:
                ideal_target_action = n_actions - 1
            else:
                ideal_target_action = 0
            k_reward = 0.7
            reward = np.exp(-k_reward * abs(action_taken - ideal_target_action))
            fl_rewards.append(float(reward) if np.isfinite(reward) else 0.0)
            try:
                fl_manager.allocator.finish_episode()
            except Exception:
                pass
            if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
                rewards_to_avg = fl_rewards[-100:]
                avg_rew = np.mean(rewards_to_avg) if rewards_to_avg else 0.0
                print(f"  FL Manager Episode {episode+1}, Avg Reward (last {len(rewards_to_avg)}): {avg_rew:.4f}")
        fl_overall_mean = np.mean(fl_rewards) if fl_rewards else 0.0
        fl_overall_std = np.std(fl_rewards) if fl_rewards else 0.0
        print(f"--- FL Trained Manager (LSTM + PPO) Finished ---")
        print(f"  Overall Mean Reward = {fl_overall_mean:.4f}, Std Dev = {fl_overall_std:.4f}")
    else:
        print("\nSkipping experiment with FL-trained predictor as FL failed.")

if __name__ == "__main__":
    main()
