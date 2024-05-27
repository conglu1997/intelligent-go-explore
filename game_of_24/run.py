import os
import openai
import backoff

client = openai.OpenAI()

from game_of_24.third_party.game24 import Game24Task
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import argparse
import numpy as np

PRINT_LLM_DEBUG = False


# Ensures that states are uniquely represented.
def convert_to_sorted_tuple(x):
    return tuple(sorted([int(i) for i in x]))


# Helper variables for actions, turn this problem into an MDP.
select_2_from_4 = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
]

select_2_from_3 = [
    [0, 1],
    [0, 2],
    [1, 2],
]

select_2_from_2 = [
    [0, 1],
]

operations = ['+', '-', '*', '/']

action_list = {
    2: [(x, y) for x in select_2_from_2 for y in operations],
    3: [(x, y) for x in select_2_from_3 for y in operations],
    4: [(x, y) for x in select_2_from_4 for y in operations],
}


def perform_operation(smaller_num, larger_num, operation):
    if operation == '+':
        new_num = larger_num + smaller_num
    elif operation == '-':
        new_num = larger_num - smaller_num
    elif operation == '*':
        new_num = larger_num * smaller_num
    elif operation == '/':
        # filter for division by 0 and non-integer division
        if smaller_num == 0 or larger_num % smaller_num != 0:
            return None
        new_num = int(larger_num / smaller_num)
    else:
        new_num = None
    return new_num


def transition_function(current_state, indices, operation):
    remaining = [current_state[i] for i in range(len(current_state)) if i not in indices]
    new_num = perform_operation(current_state[indices[0]], current_state[indices[1]], operation)
    if new_num is None:
        return None
    else:
        return convert_to_sorted_tuple(remaining + [new_num])


# Prompts for IGE.
env_description = '''You are given 4 numbers and must use basic arithmetic operations (+ - * /) to obtain 24.
At each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
A correct answer is one that uses each input exactly once and no other numbers.
Reaching 24 before the last step does not count as a correct answer.
Follow the convention that division is integer division, and never by zero.
Some examples of correct reasoning traces are as follows:
Initial state: (4 4 6 8)
Steps:
4 + 8 = 12. Next: (4 6 12)
6 - 4 = 2. Next: (2 12)
2 * 12 = 24. Next: (24)
Answer: (6 - 4) * (4 + 8) = 24
Initial state: (2 9 10 12)
Steps:
12 * 2 = 24. Next: (9 10 24)
10 - 9 = 1. Next: (1 24)
24 * 1 = 24. Next: (24)
Answer: (12 * 2) * (10 - 9) = 24
Initial state: (4 9 10 13)
Steps:
13 - 10 = 3. Next: (3 4 9)
9 - 3 = 6. Next: (4 6)
4 * 6 = 24. Next: (24)
Answer: 4 * (9 - (13 - 10)) = 24'''

json_instruction = '''Reply concisely and exactly with the following JSON format:
{"choice": X}
where X is the index of the desired choice.'''

react_json_instruction = '''First briefly reason about your plan.
Reply concisely and exactly with the following JSON format:
{"thought": X, "choice": Y}
where X is your reasoning and Y is the index of the desired choice. Make sure Y is a parsable integer!!!'''

ige_system_message = env_description + '\n\n' + '''You will be prompted to perform systematic exploration in the style of Go-Explore.
An archive will be maintained of interesting states found.
You will be prompted to:
- Select a state from the archive that is the most promising, i.e., likely to lead to a solution or more novel states.
- Explore from states intelligently, by picking new actions.
- For each new state, you will be asked to decide if the state is interestingly new and should be added to the archive.''' + '\n\n' + json_instruction

naive_llm_system_message = env_description + '\n\n' + '''You will be prompted to select an action to take at each step.''' + '\n\n' + json_instruction

react_system_message = env_description + '\n\n' + '''You will be prompted to reason about your plan and then select an action to take at each step.''' + '\n\n' + react_json_instruction


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_index_from_gpt(
        msg,
        model,
        system_message,
        print_response=PRINT_LLM_DEBUG,
        msg_history=[],
        chat_history_len=5,
):
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": system_message + f'\nYou will be given the last {chat_history_len} steps of history.'},
            *new_msg_history[-(2 * chat_history_len + 1):]
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
    print_prompts = PRINT_LLM_DEBUG

    def __init__(self):
        self.storage = []
        self.actions_left_to_take = {}
        self.total_cost = 0

    def select_new_state(self):
        raise NotImplementedError

    def select_action_(self, current_state):
        raise NotImplementedError

    def select_action(self, current_state):
        indices, operation = self.select_action_(current_state)
        self.actions_left_to_take[current_state] = [r for r in self.actions_left_to_take[current_state] if
                                                    r != (indices, operation)]
        return indices, operation

    def add_to_archive(self, current_state):
        return True

    def observe(self, current_state):
        # observe if not terminal
        if len(current_state) > 1:
            if current_state not in self.storage and (len(self.storage) == 0 or self.add_to_archive(current_state)):
                self.storage.append(current_state)
            if current_state not in self.actions_left_to_take:
                self.actions_left_to_take[current_state] = action_list[len(current_state)]

    def skip_state(self, current_state):
        raise NotImplementedError


class RandomAgent(Agent):
    def select_new_state(self):
        return random.choice(self.storage)

    def select_action_(self, current_state):
        return random.choice(action_list[len(current_state)])

    def skip_state(self, current_state):
        return False

    # override observe to not add to storage
    def observe(self, current_state):
        if len(current_state) > 1:
            if len(self.storage) == 0 and current_state not in self.storage:
                self.storage.append(current_state)
            if current_state not in self.actions_left_to_take:
                self.actions_left_to_take[current_state] = action_list[len(current_state)]


class DFSAgent(Agent):
    def select_new_state(self):
        return [s for s in self.storage if len(self.actions_left_to_take[s]) > 0][-1]

    def select_action_(self, current_state):
        return self.actions_left_to_take[current_state][0]

    def skip_state(self, current_state):
        return len(self.actions_left_to_take[current_state]) == 0


class BFSAgent(Agent):
    def select_new_state(self):
        return [s for s in self.storage if len(self.actions_left_to_take[s]) > 0][0]

    def select_action_(self, current_state):
        return self.actions_left_to_take[current_state][0]

    def skip_state(self, current_state):
        return True


class NaiveLLM(RandomAgent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.msg_history = []
        self.system_message = naive_llm_system_message
        self.chat_history_len = 3

    def select_action_(self, current_state):
        valid_actions = action_list[len(current_state)]
        choice_string = """Current state: ({input})\nPossible next steps:\n{next_steps}"""
        input = ' '.join([str(s) for s in current_state])
        converted_valid_actions = []
        for j, act in enumerate(valid_actions):
            indices, operation = act
            num_1 = current_state[indices[0]]
            num_2 = current_state[indices[1]]
            result = perform_operation(num_1, num_2, operation)
            # don't show invalid ops
            if result is None:
                continue
            # allow both orders
            if operation in ['+', '*']:
                if random.choice([True, False]):
                    num_1, num_2 = num_2, num_1
            next_state = transition_function(current_state, indices, operation)
            converted_valid_actions.append(
                (f'{num_2} {operation} {num_1} = {result}. Next: ({" ".join([str(s) for s in next_state])})', j))
        # label with indices
        converted_valid_actions = [(f'{i + 1}. {act[0]}', act[1]) for i, act in enumerate(converted_valid_actions)]
        next_steps = '\n'.join([str(act[0]) for act in converted_valid_actions])
        choice_string = choice_string.format(input=input, next_steps=next_steps)
        choice_string += f'\nChoose the best option between 1 and {len(converted_valid_actions)}'
        try:
            idx, cost, self.msg_history = get_index_from_gpt(
                choice_string, self.model, msg_history=self.msg_history,
                print_response=self.print_prompts, system_message=self.system_message,
                chat_history_len=self.chat_history_len)
            idx = int(idx) - 1
            self.total_cost += cost
            converted_idx = converted_valid_actions[idx][1]
            return valid_actions[converted_idx]
        except Exception as e:
            print(f'Action error: {e}')
            return random.choice(valid_actions)

    # clear out the chat history on choosing a new state
    def select_new_state(self):
        self.msg_history = []
        return random.choice(self.storage)


class ReactLLM(NaiveLLM):
    def __init__(self, model):
        super().__init__(model)
        self.system_message = react_system_message


class Reflexion(RandomAgent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.msg_history = []
        self.system_message = react_system_message
        self.chat_history_len = 6
        self.num_tries = 0

    def select_action_(self, current_state):
        valid_actions = action_list[len(current_state)]
        if self.num_tries >= 2 and len(current_state) == 4:
            choice_string = """Did not reach 24. Think carefully about where you went wrong in your latest attempt, and given your insights from previous attempts, try to solve the problem."""
            choice_string += """Current state: ({input})\nPossible next steps:\n{next_steps}"""
        else:
            choice_string = """Current state: ({input})\nPossible next steps:\n{next_steps}"""
        input = ' '.join([str(s) for s in current_state])
        converted_valid_actions = []
        for j, act in enumerate(valid_actions):
            indices, operation = act
            num_1 = current_state[indices[0]]
            num_2 = current_state[indices[1]]
            result = perform_operation(num_1, num_2, operation)
            # don't show invalid ops
            if result is None:
                continue
            # allow both orders
            if operation in ['+', '*']:
                if random.choice([True, False]):
                    num_1, num_2 = num_2, num_1
            next_state = transition_function(current_state, indices, operation)
            converted_valid_actions.append(
                (f'{num_2} {operation} {num_1} = {result}. Next: ({" ".join([str(s) for s in next_state])})', j))
        # label with indices
        converted_valid_actions = [(f'{i + 1}. {act[0]}', act[1]) for i, act in enumerate(converted_valid_actions)]
        next_steps = '\n'.join([str(act[0]) for act in converted_valid_actions])
        choice_string = choice_string.format(input=input, next_steps=next_steps)
        choice_string += f'\nChoose the best option between 1 and {len(converted_valid_actions)}'
        try:
            idx, cost, self.msg_history = get_index_from_gpt(
                choice_string, self.model, msg_history=self.msg_history,
                print_response=self.print_prompts, system_message=self.system_message,
                chat_history_len=self.chat_history_len)
            idx = int(idx) - 1
            self.total_cost += cost
            converted_idx = converted_valid_actions[idx][1]
            return valid_actions[converted_idx]
        except Exception as e:
            print(f'Action error: {e}')
            return random.choice(valid_actions)

    def select_new_state(self):
        self.num_tries += 1
        return random.choice(self.storage)


class IGE(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.first_state = True
        self.msg_history = []
        self.ban_list = []
        self.system_message = ige_system_message

    def get_valid_states(self):
        valid_states = [s for s in self.storage if len(self.actions_left_to_take[s]) > 0]
        if len(valid_states) == 0:
            # soft reset
            print('Error: No valid states left. Resetting...')
            valid_states = self.storage
            self.actions_left_to_take = {s: action_list[len(s)] for s in self.storage}
        return valid_states

    def select_new_state(self):
        valid_states = self.get_valid_states()
        valid_states = sorted(valid_states, key=len, reverse=True)
        if self.first_state:
            choice_string = """Possible states:\n{states}"""
            self.first_state = False
        else:
            choice_string = """Did not reach 24. Possible states:\n{states}"""
        converted_valid_states = []
        for state in valid_states:
            converted_valid_states.append(f"({' '.join([str(s) for s in state])})")
        converted_valid_states = [f'{i + 1}. {state}' for i, state in enumerate(converted_valid_states)]
        states = '\n'.join([str(state) for state in converted_valid_states])
        choice_string = choice_string.format(states=states)
        choice_string += f'\nChoose the best option between 1 and {len(converted_valid_states)}'
        try:
            idx, cost, self.msg_history = get_index_from_gpt(
                choice_string, self.model, msg_history=self.msg_history,
                print_response=self.print_prompts, system_message=self.system_message)
            idx = int(idx) - 1
            self.total_cost += cost
            return valid_states[idx]
        except Exception as e:
            print(f'State selection error: {e}')
            return random.choice(valid_states)

    def select_action_(self, current_state):
        valid_actions = self.actions_left_to_take[current_state]
        choice_string = """Current state: ({input})\nPossible next steps:\n{next_steps}"""
        input = ' '.join([str(s) for s in current_state])
        converted_valid_actions = []
        for j, act in enumerate(valid_actions):
            indices, operation = act
            num_1 = current_state[indices[0]]
            num_2 = current_state[indices[1]]
            result = perform_operation(num_1, num_2, operation)
            # don't show invalid ops
            if result is None:
                continue
            # allow both orders
            if operation in ['+', '*']:
                if random.choice([True, False]):
                    num_1, num_2 = num_2, num_1
            next_state = transition_function(current_state, indices, operation)
            converted_valid_actions.append(
                (f'{num_2} {operation} {num_1} = {result}. Next: ({" ".join([str(s) for s in next_state])})', j))
        # If no valid actions, return random option.
        if len(converted_valid_actions) == 0:
            return random.choice(valid_actions)
        # label with indices
        converted_valid_actions = [(f'{i + 1}. {act[0]}', act[1]) for i, act in enumerate(converted_valid_actions)]
        next_steps = '\n'.join([str(act[0]) for act in converted_valid_actions])
        choice_string = choice_string.format(input=input, next_steps=next_steps)
        choice_string += f'\nChoose the best option between 1 and {len(converted_valid_actions)}'
        try:
            idx, cost, self.msg_history = get_index_from_gpt(
                choice_string, self.model, msg_history=self.msg_history,
                print_response=self.print_prompts, system_message=self.system_message)
            idx = int(idx) - 1
            self.total_cost += cost
            converted_idx = converted_valid_actions[idx][1]
            return valid_actions[converted_idx]
        except Exception as e:
            print(f'Action error: {e}')
            return random.choice(valid_actions)

    def skip_state(self, current_state):
        return len(self.actions_left_to_take[current_state]) == 0

    def add_to_archive(self, current_state):
        if current_state in self.ban_list:
            return False

        valid_states = self.get_valid_states()
        choice_string = 'Current archive:\n'
        for i, state in enumerate(valid_states):
            choice_string += f'{i + 1}. ({" ".join([str(s) for s in state])})\n'
        choice_string += 'New state:\n'
        choice_string += f'({" ".join([str(s) for s in current_state])})\n'
        choice_string += 'Is this state interestingly new and should be added to the archive? Only reject if you are sure it cannot lead to a solution.\n'
        choice_string += "Choices:\n0. Don't Add\n1. Add\n"

        try:
            idx, cost, self.msg_history = get_index_from_gpt(
                choice_string, self.model, msg_history=self.msg_history,
                print_response=self.print_prompts, system_message=self.system_message)
            self.total_cost += cost
            idx = int(idx)
            if idx < 0 or idx >= 2:
                raise ValueError("Invalid archive choice.")
            if idx == 1:
                return True
            else:
                self.ban_list.append(current_state)
                return False
        except Exception as e:
            print(f'Archive filtering error: {e}')
            return False


def run_trial(init_state, target_number, max_actions, agent_class, model):
    if 'model' in agent_class.__init__.__code__.co_varnames:
        agent = agent_class(model)
    else:
        agent = agent_class()
    actions_tried = 0
    initial_state = init_state.split()  # split the input string into a list of numbers
    initial_state = convert_to_sorted_tuple(initial_state)
    agent.observe(initial_state)
    num_states_in_archive = [len(agent.storage)]
    # print(f'Input: {initial_state}')
    while actions_tried < max_actions:
        # select a state to explore from
        current_state = agent.select_new_state()
        # mutate until end reached
        while True:
            actions_tried += 1
            # transition to next state
            indices, operation = agent.select_action(current_state)
            current_state = transition_function(current_state, indices, operation)
            if current_state is None:
                break
            agent.observe(current_state)
            num_states_in_archive.append(len(agent.storage))

            # add option to early break
            if len(current_state) == 1 or agent.skip_state(current_state):
                break

        # if current_state is [target_number], success
        if current_state == (target_number,):
            return actions_tried, agent.total_cost, num_states_in_archive
    return None, agent.total_cost, num_states_in_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_idx', type=int, default=900)
    parser.add_argument('--num_problems', type=int, default=100)
    parser.add_argument('--max_actions', type=int, default=700)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='time_to_success.json')
    parser.add_argument('--model', type=str,
                        default='gpt-4-turbo-2024-04-09',
                        choices=['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13'])
    args = parser.parse_args()

    if args.debug_mode:
        args.max_actions = 200
        args.num_problems = 5

    max_workers = os.cpu_count() if args.multiprocessing else 1
    print(f'Using {max_workers} workers')
    executor = ProcessPoolExecutor(max_workers=max_workers)

    task = Game24Task()
    target_number = 24
    time_to_success = defaultdict(list)
    archive_stats = defaultdict(list)

    for selection_type, agent in [
        ('Random Agent', RandomAgent),
        ('Naive LLM', NaiveLLM),
        ('ReAct', ReactLLM),
        ('Reflexion', Reflexion),
        ('DFS', DFSAgent),
        ('BFS', BFSAgent),
        ('Intelligent Go-Explore', IGE),
    ]:
        total_cost = 0
        print(selection_type)

        inputs = [task.get_input(idx=idx) for idx in range(args.min_idx, args.min_idx + args.num_problems)]
        inputs = [(x, target_number, args.max_actions, agent, args.model) for x in inputs]
        results = executor.map(run_trial, *zip(*inputs))

        for result in tqdm.tqdm(results, total=args.num_problems):
            if result[0] is not None:
                time_to_success[selection_type].append(result[0])
                if result[1] != 0:
                    total_cost += result[1]
                    print(f'Total cost: ${total_cost:.2f}')
                archive_stats[selection_type].append(result[2])

    # print archive stats
    for k, v in archive_stats.items():
        v = [max(vv) for vv in v]
        v = np.array(v)
        print(f'{k} archive stats: mean={v.mean():.1f}, std={v.std():.1f}')

    # plot success rate v.s. number of operations tried
    for k, v in time_to_success.items():
        sorted_time_to_success = sorted(v)
        success_rate = []
        for i in range(len(sorted_time_to_success)):
            success_rate.append((i + 1) / args.num_problems)
        plt.plot(sorted_time_to_success, success_rate, label=k)
    plt.xlabel('Operations tried (log-scale)')
    plt.ylabel('Success rate')
    plt.title('Success rate v.s. number of operations tried')
    plt.ylim(0, 1)
    plt.xlim(0, args.max_actions)
    plt.legend()
    plt.savefig('success_rate.png', dpi=300)
    plt.show()

    # save results
    with open(args.save_path, 'w') as f:
        json.dump(time_to_success, f)

    with open('archive_stats.json', 'w') as f:
        json.dump(archive_stats, f)


if __name__ == '__main__':
    main()
