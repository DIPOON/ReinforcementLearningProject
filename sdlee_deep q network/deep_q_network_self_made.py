# 코드는 사실상 https://gist.github.com/ortegadn/515c7aa464d0874c62eb27191cf0b764 여기서 무지성으로 받아썼다.


from minesweeper_env import *
from replay_buffer import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.network = nn.Sequential()
        # 일단 수업시간에 다룬 DQN figure대로
        self.network.add_module('ConvolutionLayer1',
                                nn.Conv1d(in_channels=self.state_shape[0], out_channels=16, kernel_size=3))
        self.network.add_module('ReLU1', nn.ReLU())
        self.network.add_module('ConvolutionLayer2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3))
        self.network.add_module('ReLU2', nn.ReLU())
        self.network.add_module('Flatten', nn.Flatten())
        # in feature 계산 못하겠어서 일단 state_dim 좀 보려고 대충 256 넣어봄
        self.network.add_module('LinearLayer1', nn.Linear(in_features=256, out_features=128))
        self.network.add_module('LinearLayer2', nn.Linear(in_features=128, out_features=self.n_actions))

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        print(type(state_t))
        print(state_t[0, ..., 0])
        # 여기 conv에서 차원 안맞아서 수정중
        qvalues = self.network(torch.flatten(state_t))

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        #assert len(qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): # 함수는 그 자체로 완성되어야 한다.
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute V*(next_states) using predicted next q-values
    with torch.no_grad():
        next_state_values = torch.max(predicted_next_qvalues, dim=1).values

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * is_not_done * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)

"""
def make_env(clip_rewards=True, seed=None):
    # gym으로 새로운 env 만들 것 아니니까 그냥 고정
    env = MinesweeperEnv(9, 9, 10)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env
"""

def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for i in range(0, n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]

        next_s, r, done, _ = env.step(action)

        exp_replay.add(s, action, r, next_s, done)

        s = next_s
        sum_rewards += r
        if done:
            s = env.reset()

    return sum_rewards, s
"""
def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env
"""

# 버퍼가 비어있으면 안돼
env = MinesweeperEnv(9, 9, 10)
state = env.reset()
exp_replay = ReplayBuffer(10**4)
# env.observation_space.shape 대신
state_shape = [81]
# n_action 잘 몰라서 81로 했다. 0부터 80까지니까
n_actions = 81
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
for i in range(100):
    '''
    위험하지만 ram 검사를 포기한다!
    if not utils.is_enough_ram(min_available_gb=0.1):
        print("""
            Less than 100 Mb RAM available. 
            Make sure the buffer size in not too huge.
            Also check, maybe other processes consume RAM heavily.
            """
             )
        break
    '''
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    if len(exp_replay) == 10**4:
        break
print(len(exp_replay))

# Setting
batch_size = 16
obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
target_network = DQNAgent(state_shape, n_actions).to(device)
max_grad_norm = 50
opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
mean_rw_history = []

# Let's train 일단 한 게임 학습해보시죠
loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, agent, target_network)
loss.backward()
grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
opt.step()
opt.zero_grad()
"""
mean_rw_history.append(evaluate(
            make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True)
        )

plt.figure(figsize=[16, 9])
plt.subplot(2, 2, 1)
plt.title("Mean reward per life")
plt.plot(mean_rw_history)
plt.grid()
assert not np.isnan(td_loss_history[-1])
plt.subplot(2, 2, 2)
plt.title("TD loss history (smoothened)")
plt.plot(utils.smoothen(td_loss_history))
plt.grid()
plt.subplot(2, 2, 3)
plt.title("Initial state V")
plt.plot(initial_state_v_history)
plt.grid()
plt.subplot(2, 2, 4)
plt.title("Grad norm history (smoothened)")
plt.plot(utils.smoothen(grad_norm_history))
plt.grid()
plt.show()
"""