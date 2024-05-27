import os
import openai
import backoff
import json

client = openai.OpenAI()
import gym
import babyai_text
import numpy as np
from collections import deque, defaultdict, OrderedDict
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import tqdm
import argparse
import pickle
import copy

PRINT_LLM_DEBUG = False

# Environment helper variables.
# First 2 environments have horizon=64, second 3 have horizon=128
env_types = ['goto', 'pickup', 'putnext', 'open', 'pick up seq go to']


# Task type is fixed upon creation of the environment, ensure we have one of each type
def separate_types(env_name):
    envs = {key: None for key in env_types}

    num_envs = len(env_types)
    num_envs_so_far = 0

    for _ in range(1000):
        env = gym.make(env_name)
        action_kind = env.env.action_kinds[0]

        if action_kind in envs and envs[action_kind] is None:
            envs[action_kind] = env
            num_envs_so_far += 1
            if num_envs_so_far == num_envs:
                break

    assert num_envs_so_far == num_envs

    return envs


action_list = {
    0: "turn left",
    1: "turn right",
    2: "go forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
}

num_actions = len(action_list)


# Based on "Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning."
def generate_prompt(goal, deque_obs, deque_actions, timestep):
    ldo = len(deque_obs)
    lda = len(deque_actions)
    g = f'Goal of the agent: {goal}.\n'
    obs = ''
    for i in range(ldo):
        obs += f'Observation {i + timestep + 1 - ldo}: ' + ', '.join(deque_obs[i]['descriptions']) + '.\n'
        obs += f'Action {i + timestep + 1 - ldo}: '
        if i < lda:
            obs += f'{deque_actions[i]}\n'
    return g + obs


# Prompts for IGE.
env_description = '''You are an agent in an 8x8 partially-observable 2D text-based environment.
You see the 6x6 grid in front of you, and can face north, south, east, or west.
The possible actions are turn left, turn right, go forward, pick up, drop, and toggle.
At each turn, you will receive a history of the last observations and actions.
Your aim is to complete the task described in the goal.
Each tile in the grid can only contain at most one object.
Objects cannot be crossed, and may need to be bypassed or moved.
You can only move onto an empty tile or on a tile containing an open door.
You can only hold one object at a time, using pick up when they are one step in front.
Objects are dropped one tile in front and cannot be dropped when there is another object in front.
Doors are unlocked with keys of the same color using the toggle action.
Actions are deterministic, do not repeat actions if they have no effect.
You have #HORIZON# steps to complete the task.'''

json_instruction = '''Reply concisely and exactly with the following JSON format:
{"choice": X}
where X is the index of the desired choice.'''

react_json_instruction = '''First, briefly reason about your plan.
Reply concisely and exactly with the following JSON format:
{"thought": X, "choice": Y}
where X is your reasoning and Y is the index of the desired choice. Make sure Y is a parsable integer!!!'''

default_system_message = env_description + '\n\n' + '''You will be prompted at each turn to choose actions.''' + \
                         '\n\n' + json_instruction

react_system_message = env_description + '\n\n' + '''You will be prompted at each turn to first reason about your plan and then choose actions.''' + \
                       '\n\n' + react_json_instruction

react_ige_system_message = env_description + '\n\n' + '''You will be prompted to perform systematic exploration in the style of Go-Explore.
An archive will be maintained of interesting states found.
You will be prompted to first reason about your plan and then:
- Select a state from the archive that is the most promising, i.e., likely to lead to a solution or more novel states.
- Explore from states intelligently, by picking new actions.
- For each new state, you will be asked to decide if the state is interestingly new and should be added to the archive.''' + \
                           '\n\n' + react_json_instruction


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_index_from_gpt(
        msg,
        model,
        system_message,
        print_response=PRINT_LLM_DEBUG,
        msg_history=[],
        chat_history_len=3,
):
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history[- (2 * chat_history_len + 1):],
        ],
        temperature=0.7, max_tokens=1000, n=1, stop=None, response_format={"type": "json_object"}
    )
    if model == "gpt-4-turbo-2024-04-09":
        cost = response.usage.completion_tokens / 1000000 * 30 + response.usage.prompt_tokens / 1000000 * 10
    elif model == "gpt-3.5-turbo-0125":
        cost = response.usage.completion_tokens / 1000000 * 1.5 + response.usage.prompt_tokens / 1000000 * 0.5
    elif model == "gpt-4o-2024-05-13":
        cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    else:
        raise ValueError(f"Unknown model: {model}")
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    if print_response:
        print()
        print('*' * 20 + 'START')
        for j, msg in enumerate(new_msg_history[-(2 * chat_history_len + 1):]):
            stripped_msg = msg["content"].replace('\n', ' ')
            print(f'{j}, {msg["role"]}: {stripped_msg[:40]}...{stripped_msg[-40:]}')
        print(content)
        print('*' * 20 + 'END')
        print()
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    return json_dict['choice'], cost, new_msg_history


class Agent:
    def __init__(self, horizon, len_history=5, filter_actions=False):
        self.goal = None
        self.obs_queue = deque([], maxlen=len_history)
        self.acts_queue = deque([], maxlen=len_history - 1)
        self.overall_acts_queue = []
        self.total_cost = 0
        self.timestep = -1
        self.archive = OrderedDict()
        self.horizon = horizon
        self.filter_actions = filter_actions
        self.remaining_actions = OrderedDict()
        self.llm_msg_history = []

    def act_impl(self, obs, remaining_actions):
        raise NotImplementedError

    def act(self, obs, env):
        env_id = self.env_to_id(env)
        remaining_actions = self.remaining_actions[env_id]
        if len(remaining_actions) == 0 or not self.filter_actions:
            remaining_actions = list(range(num_actions))
        action = self.act_impl(obs, remaining_actions)
        self.remaining_actions[env_id] = [a for a in remaining_actions if a != action]
        self.acts_queue.append(action_list[action])
        self.overall_acts_queue.append(action)
        return action

    def add_to_archive(self, infos):
        return True

    def env_to_id(self, env):
        grid = env.grid.encode().flatten()
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        carrying = env.carrying
        return (str(grid), str(agent_pos), str(agent_dir), str(carrying))

    def get_filtered_states(self):
        filtered_choices = [v for k, v in self.archive.items() if len(self.remaining_actions[k]) > 0]
        # failsafe in case all states have actions exhausted
        if not self.filter_actions or len(filtered_choices) == 0:
            return list(self.archive.values())
        return filtered_choices

    def observe(self, obs, infos, env, no_mod=False):
        self.timestep += 1
        self.goal = obs['mission']
        self.obs_queue.append(infos)

        if no_mod:
            return

        env_id = self.env_to_id(env)

        if env_id in self.archive:
            # increment the visitation counter
            self.archive[env_id] = (self.archive[env_id][0], self.archive[env_id][1], self.archive[env_id][2],
                                    self.archive[env_id][3], self.archive[env_id][4] + 1)

            # always update if more efficient path found to the same state
            if env.step_count < self.archive[env_id][3]:
                self.archive[env_id] = (obs, infos, copy.deepcopy(self.overall_acts_queue), env.step_count,
                                        self.archive[env_id][4])
            return

        self.remaining_actions[env_id] = list(range(num_actions))

        # add to archive as a new state
        if len(self.archive) < 1 or (env_id not in self.archive and self.add_to_archive(infos)):
            self.archive[env_id] = (obs, infos, copy.deepcopy(self.overall_acts_queue), env.step_count, 1)

    def choose_new_state(self):
        return next(iter(self.archive.values()))

    def reset_state(self):
        # by default select the first state
        self.obs_queue.clear()
        self.acts_queue.clear()
        self.timestep = -1
        chosen_state = copy.deepcopy(self.choose_new_state())
        self.overall_acts_queue = copy.deepcopy(chosen_state[2])
        for act in chosen_state[2]:
            self.acts_queue.append(action_list[act])
        return chosen_state


class RandomAgent(Agent):
    def act_impl(self, obs, remaining_actions):
        action = np.random.choice(remaining_actions)
        return action


class NaiveLLM(Agent):
    def __init__(self, horizon, model, filter_actions=False, react=False):
        super().__init__(horizon, filter_actions=filter_actions)
        self.model = model
        if react:
            self.system_message = react_system_message.replace('#HORIZON#', str(horizon))
        else:
            self.system_message = default_system_message.replace('#HORIZON#', str(horizon))

    def act_impl(self, obs, remaining_actions):
        try:
            prompt = generate_prompt(goal=self.goal, deque_obs=self.obs_queue, deque_actions=self.acts_queue,
                                     timestep=self.timestep)
            if self.filter_actions:
                # map remaining actions to 0 to N continuous and then back to original action
                filtered_actions = {i: action_list[remaining_actions[i]] for i in range(len(remaining_actions))}
                prompt += '\n\nNext action choices (unexplored actions shown):\n'
                for i, action in filtered_actions.items():
                    prompt += f'{i}. {action}\n'
                prompt += f'Select an action index between 0 and {len(remaining_actions) - 1}:\n'
                action, cost, self.llm_msg_history = get_index_from_gpt(
                    prompt, self.model,
                    system_message=self.system_message,
                    msg_history=self.llm_msg_history)
                self.total_cost += cost
                action = int(action)
                if action < 0 or action >= len(remaining_actions):
                    raise ValueError("Invalid action index.")
                action = remaining_actions[int(action)]
            else:
                prompt += '\n\nNext action choices:\n'
                for i, action in enumerate(action_list.values()):
                    prompt += f'{i}. {action}\n'
                prompt += 'Select an action index between 0 and 5:\n'
                action, cost, self.llm_msg_history = get_index_from_gpt(
                    prompt, self.model,
                    system_message=self.system_message,
                    msg_history=self.llm_msg_history)
                self.total_cost += cost
                action = int(action)
                if int(action) < 0 or int(action) >= num_actions:
                    raise ValueError("Invalid action index.")
            return action
        except Exception as e:
            print(f'Error: {e}')
            # clear the last two entries in the history
            self.llm_msg_history = self.llm_msg_history[:-2]
            return np.random.choice(remaining_actions)


class IGE(NaiveLLM):
    def __init__(self, horizon, model, filter_actions=True):
        super().__init__(horizon, model, filter_actions=filter_actions, react=True)
        self.system_message = react_ige_system_message.replace('#HORIZON#', str(horizon))
        self.ban_list = []

    def choose_new_state(self):
        choices = self.get_filtered_states()
        prompt = f'Goal of the agent: {self.goal}.\nCurrent state archive:\n'
        for i, (obs, infos, _, step_count, _) in enumerate(choices):
            prompt += f'{i}. Timestep {step_count}: {", ".join(infos["descriptions"])}.\n'
        prompt += f'Select a state index between 0 and {len(choices) - 1}:\n'
        try:
            idx, cost, self.llm_msg_history = get_index_from_gpt(
                prompt, self.model,
                system_message=self.system_message,
                msg_history=self.llm_msg_history)
            self.total_cost += cost
            idx = int(idx)
            if idx < 0 or idx >= len(choices):
                raise ValueError("Invalid state index.")
            return choices[idx]
        except Exception as e:
            # clear the last two entries in the history
            self.llm_msg_history = self.llm_msg_history[:-2]
            print(f'Error: {e}')
            return choices[np.random.randint(len(choices))]

    def add_to_archive(self, infos):
        if infos['descriptions'] in self.ban_list:
            return False
        choices = self.get_filtered_states()

        prompt = f'Goal of the agent: {self.goal}.\nCurrent archive:\n'
        for (_, old_infos, _, step_count, _) in choices:
            prompt += f'Timestep {step_count}: {", ".join(old_infos["descriptions"])}.\n'
        prompt += '\nNew state:\n'
        prompt += f'Timestep {self.timestep}: {", ".join(infos["descriptions"])}.\n'
        prompt += 'A state is interestingly new if it is relevant to the task or could lead to further stepping stones, and not close to the existing states.\n'
        prompt += 'Is this state interestingly new and should be added to the archive?\n'
        prompt += "Choices:\n0. Don't Add\n1. Add\n"
        try:
            idx, cost, self.llm_msg_history = get_index_from_gpt(
                prompt, self.model,
                system_message=self.system_message,
                msg_history=self.llm_msg_history)
            self.total_cost += cost
            idx = int(idx)
            if idx < 0 or idx >= 2:
                raise ValueError("Invalid archive choice.")
            if idx == 1:
                return True
            else:
                self.ban_list.append(infos['descriptions'])
                return False
        except Exception as e:
            # clear the last two entries in the history
            self.llm_msg_history = self.llm_msg_history[:-2]
            print(f'Error: {e}')
            return False


def evaluate_single_performance(env, seed, agent, num_trials=1, max_actions_per=1000):
    if num_trials == 'variable':
        if agent.horizon == 64:
            num_trials = 4
        elif agent.horizon == 128:
            num_trials = 2
        else:
            raise ValueError('Unknown horizon')

    success = [0] * num_trials
    total_reward = [0] * num_trials
    num_states_in_archive = [0] * num_trials

    for j in range(num_trials):
        random.seed(j * 100 + seed)
        np.random.seed(j * 100 + seed)
        env.seed(seed)

        done = False
        obs, infos = env.reset()
        agent.observe(obs, infos, env)
        if j > 0:
            original_obs, original_infos, replay_action_list, *_ = agent.reset_state()
            for action in replay_action_list:
                new_obs, reward, done, infos = env.step(action)
                obs = new_obs
                agent.observe(obs, infos, env)

        actions_taken = 0

        while not done and actions_taken < max_actions_per:
            action = agent.act(obs, env)
            new_obs, reward, done, infos = env.step(action)
            obs = new_obs
            agent.observe(obs, infos, env)

            if reward > 0:
                success[j] += 1
                total_reward[j] += reward
                # early break on success
                return success, total_reward, agent.total_cost, num_states_in_archive

            actions_taken += 1

        num_states_in_archive[j] = len(agent.archive)

    return success, total_reward, agent.total_cost, num_states_in_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BabyAI-MixedTestLocal-v0')
    parser.add_argument('--min_seed', type=int, default=0)
    parser.add_argument('--max_seed', type=int, default=25)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='babyai_text_perf.pkl')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])
    args = parser.parse_args()

    if args.debug_mode:
        global env_types
        env_types = ['open']
        args.max_seed = 1
        args.multi_processing = False

    max_workers = os.cpu_count() if args.multiprocessing else 1
    print(f'Using {max_workers} workers')
    executor = ProcessPoolExecutor(max_workers=max_workers)

    envs = separate_types(args.env_name)
    successes = defaultdict(list)
    successes_full = defaultdict(list)
    archive_sizes = defaultdict(list)

    # name, agent_type, num_trials, max_actions_per
    agent_types = [
        ('Random Agent', RandomAgent, 'variable', 1000),
        ('Naive LLM', lambda horizon: NaiveLLM(horizon, args.model, react=False), 'variable', 1000),
        ('ReAct', lambda horizon: NaiveLLM(horizon, args.model, react=True), 'variable', 1000),
        ('Intelligent Go-Explore', lambda horizon: IGE(horizon, args.model), 26, 10),
    ]

    total_cost = 0
    for agent_desc, agent_type, num_trials, max_actions_per in agent_types:
        for env_type in env_types:
            total_success = 0
            total_success_full = []
            num_states_in_archive = []
            env = envs[env_type]
            print(f'env type: {env_type}, env step limit: {env.env.max_steps}')

            inputs = [(copy.deepcopy(env), seed, agent_type(env.env.max_steps), num_trials, max_actions_per) for seed in
                      range(args.min_seed, args.max_seed)]
            results = list(executor.map(evaluate_single_performance, *zip(*inputs)))

            for success, _, cost, num_states in tqdm.tqdm(results):
                total_success += max(success)
                total_cost += cost
                total_success_full.append(success)
                num_states_in_archive.append(num_states)
                if cost > 0:
                    print(f'Total cost: {total_cost:.2f}')

            successes[agent_desc].append(total_success / (args.max_seed - args.min_seed))
            successes_full[agent_desc].append(total_success_full)
            archive_sizes[agent_desc].append(num_states_in_archive)
            print(
                f'agent type: {agent_desc}, success rate: {total_success / (args.max_seed - args.min_seed)}')
            print()

        # save as we go
        with open(args.save_path, 'wb') as f:
            pickle.dump(successes, f)
        with open(args.save_path.replace('.pkl', '_full.pkl'), 'wb') as f:
            pickle.dump(successes_full, f)
        with open(args.save_path.replace('.pkl', '_archive.pkl'), 'wb') as f:
            pickle.dump(archive_sizes, f)

    # plot bars side by side
    width = 0.10
    fig, ax = plt.subplots()
    x = np.arange(len(env_types))
    for i, (agent_desc, *_) in enumerate(agent_types):
        ax.bar(x + i * width, successes[agent_desc], width, label=agent_desc)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(env_types)
    plt.xlabel('Environment Type')
    plt.ylabel('Success Rate')
    plt.title('BabyAI Text')
    plt.ylim([0., 1.])
    plt.legend()
    plt.savefig('babyai_text_perf.png')
    plt.show()


if __name__ == '__main__':
    main()
