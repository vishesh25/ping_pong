""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as cPickle
import sys
import gym
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

sys.path.append(r"\Program Files\JetBrains\PyCharm 2021.3.3\plugins\python\helpers\pydev")
warnings.filterwarnings('ignore')

# hyper parameters
H = 400  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = cPickle.load(open('save.p', 'rb'))
else:
    model = {'W1': np.random.randn(H, D) / np.sqrt(D), 'W2': np.random.randn(H) / np.sqrt(H)}

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downs ample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinear
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def show_graph():
    # creating the line chart
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df['Episode'], df['Reward'], color='maroon')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Episodes vs Rewards")
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df['Episode'], df['Running Mean'], color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Running Mean")
    plt.title("Episodes vs Running mean")
    plt.show()


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v4")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
df = pd.DataFrame(columns=["Episode", "Reward", "Running Mean"])
# Start the timer
start = time.time()
print("Start time of the working environment: {0}".format(time.asctime(time.localtime(start))))
while True:
    if render: env.render(mode='rgb_array')

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see
    # http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring bookkeeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        # 20% of 21 (18.4)
        if running_reward > -13.7:
            end = time.time()
            SecToConvert = end - start
            MinutesGet, SecondsGet = divmod(SecToConvert, 60)
            HoursGet, MinutesGet = divmod(MinutesGet, 60)
            print("End time of the working environment: {0}".format(time.asctime(time.localtime(end))))
            print("Total escape time: {0:0.0f}hr: {1:0.0f}mm: {2:0.2f}ss".format(HoursGet, MinutesGet, SecondsGet))
            print("\033[1;31;40m Crash for number of neurons: {0}".format(int(H)))
            show_graph()
            sys.exit()

        print('Resetting env. Episode %d, Reward total was: %f. Running mean: %f' %
              (episode_number, reward_sum, running_reward))
        df = df.append({'Episode': episode_number, 'Reward': reward_sum, 'Running Mean': running_reward},
                       ignore_index=True)

        if episode_number % 100 == 0:
            cPickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
