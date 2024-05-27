import os
import openai
import backoff

client = openai.OpenAI()
import numpy as np
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import argparse
import pickle
import copy

import textworld

import re

PRINT_LLM_DEBUG = False

env_description = '''You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated maze and must find a specific object. You need to explore different rooms to find the target object.

Here are the available commands:
  look:                describe the current room.
  goal:                print the goal of this game
  inventory:           print the player's inventory
  go <dir>:            move the player north, east, south, or west. You can only go in the direction indicated with an exit or a door.
  open ...:            open a door or a container. You need to open a closed door before you want to go through it.
  drop ...:            drop an object on the floor
  take ...:            take an object that is visible. Make sure the object is visible to take.
  put ... on ...:      place an object on a supporter
  take ... from ...:   take an object from a container or a supporter
  insert ... into ...: place an object into a container
  unlock ... with ...: unlock a door or a container with a key. You need to unlock a locked door with a matched key in your inventory before you want to open it.

Tips:
- The target object might be located in a closed or locked container. 
- The adjective is useful for determining whether the key is matched with the lock (e.g. non-euclidean keycard is matched with non-euclidean safe). Make sure it is matched to unlock!
- The key required to unlock the door may be in another room or locked inside a container.
- Take the key whenever you can.
- After unlocking a locked door or container, it will remain closed. You will then need to open it.

You have #HORIZON# steps to complete the task. The command "restart" is forbidden.
'''

response_instruction = '''Reply concisely and exactly with the requested format.'''

default_system_message = env_description + '\n\n' + '''You will be prompted at each turn to choose actions.''' + \
                         '\n\n' + response_instruction

ige_system_message = env_description + '\n\n' + '''You will be prompted to perform systematic exploration in the style of Go-Explore.
An archive will be maintained of interesting states found.
You will be prompted to:
- Select a state from the archive that is likely to lead to a solution or more novel states. Note: Keep the target in mind and do not miss it.
- Explore from states intelligently, by picking new actions.
- Remove uninteresting states from archive.
''' + response_instruction + "\n\n"


def extract_action(text):
    # This pattern ensures that the "> " pattern appears at the start of the string or after a newline,
    # and captures everything until a period (if present) or the end of the line.
    action_pattern = r"> (.*?)(?:\.|$)"

    # Search for the pattern in the text
    match = re.search(action_pattern, text)

    # If a match is found, return the matching group which contains the action
    if match:
        action = match.group(1).strip()  # Strip any whitespace around the action
        return action.rstrip('.')  # Remove any trailing period
    else:
        # Return None or an appropriate message if no action is found
        return None


def extract_numbers_and_lists(text):
    # Define patterns to match integers and lists of integers after the ">"
    number_pattern = r"> (\d+)"  # Matches a number following ">"
    list_pattern = r"> (\[\s*(?:\d+\s*,\s*)*\d*\s*\])"  # Matches lists of numbers in brackets following ">", allows empty list

    # Initialize variables
    idx_choose = None
    idx_filtered = None

    # Search for the first matching number
    match_number = re.search(number_pattern, text)
    if match_number:
        idx_choose = int(match_number.group(1))

    # Search for the first matching list
    match_list = re.search(list_pattern, text)
    if match_list:
        list_str = match_list.group(1)
        # Evaluate the list string to convert it into a list
        # Ensure we convert the empty string to a proper empty list
        if list_str == "[]":
            idx_filtered = []
        else:
            idx_filtered = [int(num) for num in re.findall(r'\d+', list_str)]

    return idx_choose, idx_filtered


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_action_from_gpt(msg, model, system_message, print_response=PRINT_LLM_DEBUG, return_filter_list=False):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=0.3, max_tokens=1024, top_p=0.2, stop=None
    )
    if model == 'gpt-4o-2024-05-13':
        cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    elif model == "gpt-4-turbo-2024-04-09":
        cost = response.usage.completion_tokens / 1000000 * 30 + response.usage.prompt_tokens / 1000000 * 10
    elif model == "gpt-3.5-turbo-0125":
        cost = response.usage.completion_tokens / 1000000 * 1.5 + response.usage.prompt_tokens / 1000000 * 0.5
    else:
        raise ValueError(f"Unknown model: {model}")
    response = response.choices[0].message.content
    if print_response:
        print('-' * 10)
        print(msg)
        print(response)
        print('-' * 10)
    if not return_filter_list:
        action = extract_action(response)
        return action, cost
    else:
        idx_choose, idx_filtered = extract_numbers_and_lists(response)
        return idx_choose, idx_filtered, cost


def get_room_name(text):
    pattern = r'-= (.*?) =-'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


class Agent:
    def __init__(self, horizon, task_type, filter_actions=False):
        self.obs_queue = []
        self.acts_queue = []
        self.total_cost = 0

        self.tried_actions = {}
        self.filter_actions = filter_actions

        self.archive = OrderedDict()
        self.horizon = horizon
        self.task_type = task_type

        self.goal = ''

        self.done_archive = {}

    def act_impl(self, obs, infos):
        raise NotImplementedError

    def get_state_id(self, infos):
        inventory = infos['inventory']
        location = infos['description']
        return (location, inventory)

    def act(self, obs, infos):
        action = self.act_impl(obs, infos)
        self.acts_queue.append(action)
        state_id = self.get_state_id(infos)
        if self.filter_actions:
            if state_id in self.tried_actions and action not in self.tried_actions[state_id]:
                self.tried_actions[state_id].append(action)
            else:
                self.tried_actions[state_id] = [action]
        return action

    def add_or_update(self, obs, infos):
        inventory = infos['inventory']
        location = infos['description']
        state_id = self.get_state_id(infos)
        step_count = infos['moves']

        if 'last_command' in infos and ('lost' in obs):
            # force update with failure state
            self.archive[state_id] = ((obs, inventory), infos, copy.deepcopy((self.acts_queue, self.obs_queue)))
            self.done_archive[state_id] = (infos['last_command'], get_room_name(infos['description']))
            return

        if 'last_command' in infos and ('you unlock' in obs.lower() or 'you open' in obs.lower()):
            # force update with info unlock/open actions
            self.archive[state_id] = ((obs, inventory), infos, copy.deepcopy((self.acts_queue, self.obs_queue)))
            return

        if state_id in self.archive and step_count > self.archive[state_id][1]['moves']:
            # ignore longer path
            return
        else:
            self.archive[state_id] = ((location, inventory), infos, copy.deepcopy((self.acts_queue, self.obs_queue)))

    def init_observe(self, obs, infos, env):
        self.obs_queue = []
        self.acts_queue = []

        # remove art block
        art_block = "\n\n\n                    ________  ________  __    __  ________        \n                   |        \\|        \\|  \\  |  \\|        \\       \n                    \\$$$$$$$$| $$$$$$$$| $$  | $$ \\$$$$$$$$       \n                      | $$   | $$__     \\$$\\/  $$   | $$          \n                      | $$   | $$  \\     >$$  $$    | $$          \n                      | $$   | $$$$$    /  $$$$\\    | $$          \n                      | $$   | $$_____ |  $$ \\$$\\   | $$          \n                      | $$   | $$     \\| $$  | $$   | $$          \n                       \\$$    \\$$$$$$$$ \\$$   \\$$    \\$$          \n              __       __   ______   _______   __        _______  \n             |  \\  _  |  \\ /      \\ |       \\ |  \\      |       \\ \n             | $$ / \\ | $$|  $$$$$$\\| $$$$$$$\\| $$      | $$$$$$$\\\n             | $$/  $\\| $$| $$  | $$| $$__| $$| $$      | $$  | $$\n             | $$  $$$\\ $$| $$  | $$| $$    $$| $$      | $$  | $$\n             | $$ $$\\$$\\$$| $$  | $$| $$$$$$$\\| $$      | $$  | $$\n             | $$$$  \\$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$\n             | $$$    \\$$$ \\$$    $$| $$  | $$| $$     \\| $$    $$\n              \\$$      \\$$  \\$$$$$$  \\$$   \\$$ \\$$$$$$$$ \\$$$$$$$ \n\n"
        obs = obs.replace(art_block, '')
        self.obs_queue.append(obs)
        self.add_or_update(obs, infos)
        self.init_obs_queue = copy.deepcopy(self.obs_queue)
        self.goal = infos['objective']
        return

    def add_to_archive(self, obs, infos):
        return True

    def create_history_obs(self, input_queue=None):
        if input_queue:
            acts_queue = input_queue[0]
            obs_queue = input_queue[1]
        else:
            acts_queue = self.acts_queue
            obs_queue = self.obs_queue
        assert len(acts_queue) + 1 == len(obs_queue), 'wrong len of history queue'
        # Initialize an empty list to collect formatted strings
        formatted_lines = [obs_queue[0]]

        # Iterate over both lists using the `zip` function to pair each action with its observation
        for i, (action, observation) in enumerate(zip(acts_queue, obs_queue[1:]), 1):
            # Format each action and observation as specified and append to the list
            formatted_lines.append(f"> {action}")
            formatted_lines.append(observation)
        # Join all formatted lines into a single string with each line separated by a newline character
        result_string = "\n".join(formatted_lines)
        return result_string

    def observe(self, obs, infos, env, add):
        self.obs_queue.append(obs)

        # add to archive as a new state
        if add and self.add_to_archive(obs, infos):
            self.add_or_update(obs, infos)

    def choose_new_state(self, input_archive=None):
        return next(iter(self.archive.values())), None

    def reset_state(self):
        # by default select the first state
        self.obs_queue = []
        self.acts_queue = []

        chosen_state, state_id = self.choose_new_state()
        self.acts_queue, self.obs_queue = copy.deepcopy(chosen_state[2])
        return chosen_state


class NaiveLLM(Agent):
    def __init__(self, horizon, task_type, model, filter_actions=False, react=True):
        super().__init__(horizon, task_type, filter_actions=filter_actions)
        self.model = model
        self.react = react
        self.system_message = default_system_message.replace('#HORIZON#', str(horizon))

    def act_impl(self, obs, infos):
        state_id = self.get_state_id(infos)

        try:
            prompt = f"Interact with the textworld to solve the task.\nYour goal: "
            prompt += self.create_history_obs()
            prompt += f"\nYour current inventory: {infos['inventory']}"

            if self.filter_actions:
                prompt += f"\nMake sure your action is feasible and useful.\nHere are the actions that you have already tried: [{', '.join(self.tried_actions[state_id]) if state_id in self.tried_actions else 'None'}]."
                prompt += "Avoid repeating tried actions if there are still useful and feasible actions that have not yet been tried."
                if self.done_archive:
                    prompt += "\nPreviously, you took the wrong item and made the game fail: { "
                    for k, v in self.done_archive.items():
                        prompt += f"You [{v[0]}] in [{v[1]}]. "
                    prompt += "} PLEASE DON'T TAKE THESE ITEMS IN THESE ROOMS IN ANY CASE!!!"
                prompt += f"\nThink carefully about what are the useful and feasible actions you haven't tried in the current room. Prefer finishing exploring the current room before going to the next."

            if self.react:
                prompt += "\nFirst briefly reason it and then output the command you want in the format \'> command\'. Ensure only one command is included."
            else:
                prompt += "\nDirectly output the command in the format \'> command\'. Ensure only one command is included."

            action, cost = get_action_from_gpt(prompt, self.model, system_message=self.system_message)

            self.total_cost += cost
            assert action, "Action can not be None"
            return action

        except Exception as e:
            print(f'Error: {e}')
            return 'look'

    def choose_new_state(self):
        raise NotImplementedError


class Reflexion(NaiveLLM):
    def __init__(self, horizon, task_type, model, filter_actions=False):
        super().__init__(horizon, task_type, model, filter_actions=filter_actions)
        self.model = model
        self.system_message = default_system_message.replace('#HORIZON#', str(horizon))
        self.reflexion_queue = []
        self.reflect_mode = False

    def act_impl(self, obs, infos):

        try:
            if not self.reflect_mode:
                prompt = "Interact with the textworld to solve the task.\n"
            else:
                prompt = "You were previously solving a textworld. "
                prompt += "And here are all your past attempts:\n"
                for i in range(len(self.reflexion_queue)):
                    prompt += f"Attempt #{i + 1}:\n{self.create_history_obs(input_queue=self.reflexion_queue[i])}\n\n"
                prompt += "Think carefully about where you went wrong in your latest attempt, and given your insights from previous attempts, try to solve the problem.\n\n"
                prompt += "The current interaction: \n"

            prompt += self.create_history_obs()

            prompt += "\nFirst briefly reason about it and then output the command in the format \'> command\'. Ensure only one command is included."

            action, cost = get_action_from_gpt(prompt, self.model, system_message=self.system_message)

            self.total_cost += cost
            assert action, "Action can not be None"
            return action

        except Exception as e:
            print(f'Error: {e}')
            return 'look'

    def reset_state(self):
        self.reflect_mode = True
        self.reflexion_queue.append((self.acts_queue, self.obs_queue))

        self.obs_queue = copy.deepcopy(self.init_obs_queue)
        self.acts_queue = []

        return None


class IGE(NaiveLLM):
    def __init__(self, horizon, task_type, model, filter_actions=True):
        super().__init__(horizon, task_type, model, filter_actions=filter_actions)
        self.system_message = ige_system_message.replace('#HORIZON#', str(horizon))

    def choose_new_state(self, input_archive=None):
        if input_archive:
            keys = list(input_archive.keys())
            choices = list(input_archive.values())
        else:
            keys = list(self.archive.keys())
            choices = list(self.archive.values())

        prompt = "Choose an interesting new or promising state from the archive to continue exploring from. Imagine the maze from the different rooms.\n"
        prompt += f'Your goal: {self.goal}\n'
        prompt += "Current state archive:\n"
        for i, (obs, infos, action_obs_list) in enumerate(choices):
            prompt += f"{i}: [{infos['moves']} steps used]\nCurrent Location:\n{obs[0]}\nCurrent Inventory: {obs[1]}\nHave tried actions from this state: [{', '.join(self.tried_actions[keys[i]]) if keys[i] in self.tried_actions else 'None'}] \n\n"
        prompt += f'Choose a state where you think there are still unexplored actions. '
        prompt += f'First briefly reason about what you want to choose, then output an integer X to select a state index between 0 and {len(choices) - 1} (Format: \'Choose > X\'). '
        prompt += f'Then you need to output the redundant states you want to remove from the archive. A state is redundant if there are another state making more progress in the SAME room (e.g. have more items in inventory or unlocked/open a door).\n'
        prompt += f'Also remove the losing states.\n'
        prompt += f"First reason about every state you are going to remove. Please ONLY remove states if you think it is necessary. Definitely don't remove a state if there are no other states in the same room. "
        prompt += f'Then output the states you want to remove (Format: \'Remove > [a, b] or [a] or [] if you don\'t want to remove any\'). '
        prompt += f"Make sure you include your answer of 'Choose' and 'Remove'!"
        try:
            idx_choose, idx_filtered, cost = get_action_from_gpt(prompt, self.model, system_message=self.system_message,
                                                                 return_filter_list=True)

            self.total_cost += cost

            assert not idx_choose is None, "selection cannot be None"
            if idx_choose < 0 or idx_choose >= len(choices):
                raise ValueError("Invalid index.")
            choice = choices[idx_choose]
            key = keys[idx_choose]

            if idx_filtered:
                for idx in idx_filtered:
                    key = keys[idx]
                    del self.archive[key]
                if PRINT_LLM_DEBUG:
                    print(f"remove {idx_filtered}")

            return choice, key
        except Exception as e:
            if PRINT_LLM_DEBUG:
                print(f'Error: {e}. Manually choose state.')
            choices = list(self.archive.values())
            rand = np.random.randint(len(choices))
            return choices[rand], None


def evaluate_single_performance(env_id, agent_name, agent, num_trials=1, max_actions_per=1000):
    success = [0] * num_trials

    req_infos = textworld.EnvInfos(
        feedback=True,  # Response from the game after typing a text command.
        description=True,  # Text describing the room the player is currently in.
        inventory=True,  # Text describing the player's inventory.
        command_templates=True,
        admissible_commands=True,
        location=True,
        moves=True,
        won=True,
    )

    env = textworld.start(env_id, request_infos=req_infos)

    for j in range(num_trials):

        done = False
        game_state = env.reset()
        obs = game_state['feedback']
        if j == 0 or 'LLM' in agent_name:
            agent.init_observe(game_state['feedback'], game_state, env)
        else:
            if 'Reflexion' in agent_name:
                agent.reset_state()
            else:
                # return to state
                original_obs, original_infos, action_obs_list, *_ = agent.reset_state()
                for action in action_obs_list[0]:
                    game_state, reward, done = env.step(action)
                    obs = game_state['feedback']

        actions_taken = 0

        while not done and game_state['moves'] < agent.horizon and actions_taken < max_actions_per:
            action = agent.act(obs, game_state)
            game_state, reward, done = env.step(action)
            obs = game_state['feedback']
            if PRINT_LLM_DEBUG:
                print(obs)
            agent.observe(obs, game_state, env, add=True)
            if game_state['won']:
                success[j] += 1
                # early break on success
                print(f"{env_id} win with budget {j * max_actions_per + actions_taken} and move {game_state.moves}!")
                return success, game_state['won'], (
                    agent.total_cost, j * max_actions_per + actions_taken), agent_name, env_id
            actions_taken += 1

    print(f"{env_id} fails!")
    return success, game_state['won'], (agent.total_cost, j * max_actions_per + actions_taken), agent_name, env_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_dir', type=str, default='textworld/tw_games/treasure_hunter')
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])
    args = parser.parse_args()

    max_steps = 40
    agent_types = [
        ('Naive LLM', lambda horizon, task_type: NaiveLLM(horizon, task_type, args.model, react=False), 3, max_steps),
        ('ReAct', lambda horizon, task_type: NaiveLLM(horizon, task_type, args.model, react=True), 3, max_steps),
        ('Reflexion', lambda horizon, task_type: Reflexion(horizon, task_type, args.model), 3, max_steps),
        ('Intelligent Go-Explore', lambda horizon, task_type: IGE(horizon, task_type, args.model), 24, 5),
    ]

    env_ids = []
    with os.scandir(args.game_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".ulx"):
                env_ids.append(entry.path)

    max_workers = len(env_ids) if args.multiprocessing else 1
    print(f'Using {max_workers} workers')
    executor = ProcessPoolExecutor(max_workers=max_workers)

    for agent_name, agent_type, num_trials, max_actions_per in agent_types:
        total_num = 0
        total_success = 0
        total_cost = 0
        total_success_full = []

        inputs = [(env_id, agent_name, agent_type(max_steps, agent_name), num_trials, max_actions_per) for env_id in
                  env_ids]
        results = list(executor.map(evaluate_single_performance, *zip(*inputs)))

        for res in results:
            try:
                file_path = os.path.join(args.save_dir,
                                         f"{agent_name}/{res[-1].split('/')[-2]}/{res[-1].split('/')[-1]}.pkl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    pickle.dump(res, f)
            except Exception as e:
                print(e)
                continue

        for success, _, cost, task_type, env_id in results:
            total_num += 1
            total_success += max(success)
            total_cost += cost[0]
            total_success_full.append((success, task_type))

        print(f'Total cost: {total_cost:.2f}')
        print(f"total_num {total_num}\ntotal_success {total_success}\nRate {total_success / total_num}")


if __name__ == '__main__':
    main()
