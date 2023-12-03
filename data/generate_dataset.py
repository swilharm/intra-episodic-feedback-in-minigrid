import json
import random
import sys

from data.stats_dataset import get_stats
from envs.shared_env import SharedEnv
from speaker.heuristic_speaker import HeuristicSpeaker


def generate_dataset(split: str,
                     size: int,
                     num_distractors: int,
                     ambiguity: bool = False,
                     holdouts: bool = False) -> None:
    """
    Creates a dataset according to the parameters and saves it as a json file in data/
    :param split: can be 'train', 'val' or 'test'.
    Train sets have 1250 instances per object type, val and test sets have 50.
    :param size: The size of the map to generate
    :param num_distractors: The number of distractors (besides the target) to place
    :param ambiguity: Whether instances should be created with ambiguity
    (A distractor with same color and shape as the target)
    :param holdouts: Wether held out objects should be included
    """
    train = split == 'train'
    val = split == 'val'
    test = split == 'test'

    # seed = {train: 2, val: 3, test: 5, test_ambig: 7, test_holdout: 11} * size * num_distractors
    if train:
        random.seed(2 * size * num_distractors)
    elif val:
        random.seed(3 * size * num_distractors)
    elif test:
        if ambiguity:
            random.seed(7 * size * num_distractors)
        elif holdouts:
            random.seed(11 * size * num_distractors)
        else:
            random.seed(5 * size * num_distractors)

    color_types = [('yellow', 'ball'), ('red', 'ball'), ('green', 'ball'), ('blue', 'ball'),
                   ('purple', 'key'), ('red', 'key'), ('green', 'key'), ('grey', 'key')]
    if holdouts:
        color_types.extend([('yellow', 'key'), ('blue', 'key'), ('purple', 'ball'), ('grey', 'ball')])

    # instances_per_object = {train: 1250, val: 50, test: 50}
    if train:
        instances_per_object = 1250
    else:
        instances_per_object = 50

    object_list = color_types * instances_per_object
    random.shuffle(object_list)
    if ambiguity:
        distractor_list = color_types * (instances_per_object * (num_distractors - 1))
    else:
        distractor_list = color_types * (instances_per_object * num_distractors)
    random.shuffle(distractor_list)

    color_type_configs = []
    for _ in range(len(color_types) * instances_per_object):
        target = object_list.pop()
        distractors = []
        while len(distractors) < num_distractors:
            if ambiguity and len(distractors) == 0:
                distractor = target
            else:
                for i in range(100):
                    distractor = distractor_list.pop(0)
                    if distractor != target:
                        break
                    if i < 99:
                        distractor_list.append(distractor)
                    else:
                        print("Could not find non-ambiguous after 100 tries, proceeding with ambiguous instance")
            distractors.append(distractor)
        config = {'target': {'color': target[0], 'type': target[1]},
                  'distractors': [{'color': distractor[0], 'type': distractor[1]} for distractor in distractors]}
        color_type_configs.append(config)

    env = SharedEnv(
        size=size,
        speaker=HeuristicSpeaker,
        configs=color_type_configs,
        render_mode="rgb_array",
    )

    full_configs = []
    for _ in range(len(color_type_configs)):
        env.reset(seed=random.randrange(sys.maxsize))
        agent = {'pos': (int(env.agent_pos[0]), int(env.agent_pos[1])), 'dir': int(env.agent_dir)}
        target = {'color': env.target.color, 'type': env.target.type,
                  'pos': (int(env.target.cur_pos[0]), int(env.target.cur_pos[1]))}
        distractors = []
        for distractor in env.distractors:
            distractors.append({'color': distractor.color, 'type': distractor.type,
                                'pos': (int(distractor.cur_pos[0]), int(distractor.cur_pos[1]))})
        config = {'agent': agent, 'target': target, 'distractors': distractors}
        full_configs.append(config)

    file_name = f"data/fetch_{size}x{size}_{num_distractors}d"
    if train:
        file_name += '_train'
    elif val:
        file_name += '_val'
    else:
        file_name += '_test'
        if ambiguity:
            file_name += '_ambig'
        if holdouts:
            file_name += '_holdouts'
    file_name += '.json'
    with open(file_name, 'w') as file:
        json.dump(full_configs, file)

    targets, distractors, ambiguities = get_stats(full_configs)
    print(f"dataset: {file_name}")
    print(f"num instances: {len(full_configs)}")
    print(f"targets: {targets}")
    print(f"distractors: {distractors}")
    print(f"ambiguities: {ambiguities}")
    print()


if __name__ == "__main__":
    print()

    generate_dataset('test', 6, 2)
    generate_dataset('test', 6, 2, ambiguity=True)
    generate_dataset('test', 6, 2, holdouts=True)

    generate_dataset('train', 12, 5)
    generate_dataset('val', 12, 5)
    generate_dataset('test', 12, 5)
    generate_dataset('test', 12, 5, ambiguity=True)
    generate_dataset('test', 12, 5, holdouts=True)

    generate_dataset('test', 18, 8)
    generate_dataset('test', 18, 8, ambiguity=True)
    generate_dataset('test', 18, 8, holdouts=True)
