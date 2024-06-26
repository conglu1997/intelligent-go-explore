#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import sys
import argparse
from os.path import join as pjoin

import numpy as np

import textworld
import textworld.challenges
import textworld.render

from textworld.generator import QuestGenerationError
from textworld.generator.text_grammar import MissingTextGrammar


def exit_listing_challenges(challenge=None):
    msg = ""
    if challenge is not None:
        msg += "Unknown challenge: {}\n\n".format(challenge)

    msg += "Available challenges:\n  "
    msg += "\n  ".join(sorted(textworld.challenges.CHALLENGES))
    print(msg)
    sys.exit(1)


def _maybe_load_third_party_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--third-party")
    parser.add_argument("--silent", action="store_true")
    args, _ = parser.parse_known_args()
    if args.third_party:
        if not args.silent:
            print("Importing challenges from {}.".format(args.third_party))

        import importlib
        spec = importlib.util.spec_from_file_location("tw_custom_challenge", args.third_party)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def build_parser(default_parser_only=True):
    description = "Make a TextWorld game (.z8 or .ulx)."
    general_parser = argparse.ArgumentParser(add_help=False, description=description)
    general_parser.add_argument("--third-party", metavar="PATH",
                                help="Load an external python file. Useful to register custom challenges on-the-fly.")

    parser = argparse.ArgumentParser(parents=[general_parser])
    subparsers = parser.add_subparsers(dest="subcommand", title='Types of game to create')

    custom_parser = subparsers.add_parser("custom", help='Make a custom game.')
    group = custom_parser.add_argument_group('Custom game settings')
    group.add_argument("--world-size", type=int, default=5, metavar="SIZE",
                       help="Nb. of rooms in the world.")
    group.add_argument("--nb-objects", type=int, default=10, metavar="NB",
                       help="Minimum nb. of objects in the world.")

    cfg_group = custom_parser.add_argument_group('Grammar settings')
    cfg_group.add_argument("--theme", default="house",
                           help="Theme to use for generating the text. Default: %(default)s")
    cfg_group.add_argument("--include-adj", action="store_true",
                           help="Turn on adjectives.")
    cfg_group.add_argument("--blend-descriptions", action="store_true",
                           help="Blend descriptions across consecutive sentences.")
    cfg_group.add_argument("--ambiguous-instructions", action="store_true",
                           help="Refer to an object using its type (e.g. red container vs. red chest).")
    cfg_group.add_argument("--only-last-action", action="store_true",
                           help="Intruction only describes the last action of quest.")
    cfg_group.add_argument("--blend-instructions", action="store_true",
                           help="Blend instructions across consecutive actions.")
    cfg_group.add_argument("--entity-numbering", action="store_true",
                           help="Append a number after an entity name if there is not enough"
                                " variation for it (e.g. 'red apple 2').")

    quest_group = custom_parser.add_argument_group('Quest settings')
    quest_group.add_argument("--nb-parallel-quests", type=int, default=1,
                             help="Nb. of parallel quests the game will have. Default: %(default)s.")
    quest_group.add_argument("--quest-length", type=int, metavar="LENGTH",
                             help="Nb. of actions the quest requires to be completed. It is a shorthand"
                                  " for '--quest-min-length N --quest-max-length N --quest-max-depth N'.")
    quest_group.add_argument("--quest-breadth", type=int, metavar="BREADTH",
                             help="Nb. of subquests the quests will have. It is a shorthand"
                                  " for '--quest-min-breadth N --quest-max-breadth N'.")

    quest_group_advanced = custom_parser.add_argument_group('Quest settings (advanced)')
    quest_group_advanced.add_argument("--quest-min-length", type=int, default=1, metavar="LENGTH",
                                      help="Minimum nb. of actions the quest requires to be completed. "
                                           " This setting is ignored if --quest-length is provided."
                                           " Default: %(default)s.")
    quest_group_advanced.add_argument("--quest-max-length", type=int, default=5, metavar="LENGTH",
                                      help="Maximum nb. of actions the quest requires to be completed. "
                                           " This setting is ignored if --quest-length is provided."
                                           " Default: %(default)s.")

    quest_group_advanced.add_argument("--quest-min-breadth", type=int, default=1, metavar="BREADTH",
                                      help="Minimum nb. of subquests the quests can have. "
                                           " This setting is ignored if --quest-breadth is provided."
                                           " Default: %(default)s.")
    quest_group_advanced.add_argument("--quest-max-breadth", type=int, default=5, metavar="BREADTH",
                                      help="Maxmimum nb. of subquests the quests can have. "
                                           " This setting is ignored if --quest-breadth is provided."
                                           " Default: %(default)s.")

    quest_group_advanced.add_argument("--quest-min-depth", type=int, default=1, metavar="DEPTH",
                                      help="Minimum nb. of actions the subquests can have. Default: %(default)s.")
    quest_group_advanced.add_argument("--quest-max-depth", type=int, default=5, metavar="DEPTH",
                                      help="Maximum nb. of actions the subquests can have. "
                                           " This setting is ignored if --quest-length is provided."
                                           " Default: %(default)s.")

    def _add_general_settings(parser):
        general_group = parser.add_argument_group('General settings')
        general_group.add_argument("--output", default="./tw_games/", metavar="PATH",
                                   help="Path where to save the generated game. If it points to a folder,"
                                        " the game's UUID will be used as the filename.")
        general_group.add_argument('--seed', type=int)
        general_group.add_argument('--format', choices=["ulx", "z8"], default="z8",
                                   help="Which format to use when compiling the game. Default: %(default)s")
        general_group.add_argument("--overview", action="store_true",
                                   help="Display an overview of the generated game.")
        general_group.add_argument("--save-overview", action="store_true",
                                   help="Save the overview image of the generated game alongside the game as a PNG file.")
        general_group.add_argument("-f", "--force", action="store_true")

        verbosity_group = general_group.add_mutually_exclusive_group()
        verbosity_group.add_argument("--silent", action="store_true")
        verbosity_group.add_argument("-v", "--verbose", action="store_true")

    _add_general_settings(custom_parser)

    challenge_parsers = []
    for challenge_name, (desc, _, add_challenge_arguments) in textworld.challenges.CHALLENGES.items():
        challenge_parser = subparsers.add_parser(challenge_name, help=desc)
        add_challenge_arguments(challenge_parser)
        _add_general_settings(challenge_parser)
        challenge_parsers.append(challenge_parser)

    if default_parser_only:
        return parser

    return parser, custom_parser, challenge_parsers


if __name__ == "__main__":
    _maybe_load_third_party_module()
    parser, custom_parser, challenge_parsers = build_parser(default_parser_only=False)
    args = parser.parse_args()

    if args.subcommand is None:
        print("Need to specify which type of game to create (either 'custom' or a challenge).")
        exit_listing_challenges(args.subcommand)

    if args.seed is None:
        args.seed = np.random.randint(65635)

    if not args.silent:
        print("Global seed: {}".format(args.seed))

    options = textworld.GameOptions()
    options.seeds = args.seed
    dirname, basename = os.path.split(args.output)
    options.path = pjoin(os.path.abspath(dirname), basename)
    options.file_ext = "." + args.format
    options.force_recompile = args.force

    if args.subcommand == "custom":
        options.nb_rooms = args.world_size
        options.nb_objects = args.nb_objects
        options.nb_parallel_quests = args.nb_parallel_quests
        options.chaining.min_breadth = args.quest_min_breadth
        options.chaining.min_depth = args.quest_min_depth
        options.chaining.min_length = args.quest_min_length
        options.chaining.max_breadth = args.quest_max_breadth
        options.chaining.max_depth = args.quest_max_depth
        options.chaining.max_length = args.quest_max_length

        options.grammar.theme = args.theme
        options.grammar.include_adj = args.include_adj
        options.grammar.only_last_action = args.only_last_action
        options.grammar.blend_instructions = args.blend_instructions
        options.grammar.blend_descriptions = args.blend_descriptions
        options.grammar.ambiguous_instructions = args.ambiguous_instructions
        options.grammar.allowed_variables_numbering = args.entity_numbering

        if args.quest_length:
            options.chaining.min_length = args.quest_length
            options.chaining.max_length = args.quest_length
            options.chaining.max_depth = args.quest_length

        if args.quest_breadth:
            options.chaining.min_breadth = args.quest_breadth
            options.chaining.max_breadth = args.quest_breadth

        try:
            game_file, game = textworld.make(options)
        except QuestGenerationError:
            msg = ("No quest can be generated with the provided options:\n"
                   "\n{}\n".format(options)
                   + "\nTry relaxing the quest generation constraints via"
                     " the quest advanced settings (see 'tw-make custom --help').")
            custom_parser.error(msg)
        except MissingTextGrammar:
            msg = ("Theme '--theme {theme}' doesn't exist.\n"
                   "Check in available themes in '{path}/'."
                   ).format(theme=args.theme, path=options.kb.text_grammars_path)
            custom_parser.error(msg)

    elif args.subcommand in textworld.challenges.CHALLENGES:
        _, make_game, _ = textworld.challenges.CHALLENGES[args.subcommand]
        options.grammar.only_last_action = True
        game = make_game(settings=args.__dict__, options=options)
        game_file = textworld.generator.compile_game(game, options)
    else:
        exit_listing_challenges(args.subcommand)

    if not args.silent:
        print("Game generated: {}".format(game_file))
        print(f"Solution {len(game.metadata['walkthrough'])}")

    if args.verbose:
        print("\nObjective:")
        print(game.objective)

        if "walkthrough" in game.metadata:
            print("\nWalkthrough:")
            print(" > ".join(game.metadata["walkthrough"]))

        nb_locations = sum(1 for e in game.infos.values() if e.type == "r")
        nb_objects = sum(1 for e in game.infos.values() if e.type not in {"r", "d", "P", "I"})
        print("\n-= Stats =-")
        print("Nb. locations: {}".format(nb_locations))
        print("Nb. objects: {}".format(nb_objects))

    if args.overview:
        textworld.render.visualize(game, interactive=True)

    if args.save_overview:
        image = textworld.render.visualize(game)
        overview_file = game_file.replace("." + args.format, ".png")
        image.save(overview_file)

        if not args.silent:
            print("Overview image: {}".format(overview_file))
