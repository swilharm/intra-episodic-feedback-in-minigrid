import json
import os
from typing import List


def get_stats(dataset: List[dict]):
    """
    Calculates and returns statistics for the dataset.
    Statistics include the types of targets, types of distractors and types of ambiguities appearing in the dataset.
    :param dataset: The list of instance dicts, usually read from a json file
    :return: Target distribution, distractor distribution and ambiguity distribution
    """
    targets = {}
    distractors = {}
    num_ambiguities = {}
    for config in dataset:
        ambiguities = 0
        target = (config['target']['color'], config['target']['type'])
        if target not in targets:
            targets[target] = 0
        targets[(config['target']['color'], config['target']['type'])] += 1
        for distractor in config['distractors']:
            if (distractor['color'], distractor['type']) not in distractors:
                distractors[(distractor['color'], distractor['type'])] = 0
            distractors[(distractor['color'], distractor['type'])] += 1
            if config['target']['color'] == distractor['color'] and config['target']['type'] == distractor['type']:
                ambiguities += 1
        num_ambiguities.setdefault(ambiguities, 0)
        num_ambiguities[ambiguities] += 1

    return targets, distractors, num_ambiguities


if __name__ == "__main__":
    print()
    for dataset in [file for file in os.listdir('data') if file.endswith('.json')]:
        with open(os.path.join('data', dataset), 'r') as file:
            config = json.load(file)
        targets, distractors, ambiguities = get_stats(config)
        print(f"dataset: {dataset}")
        print(f"num instances: {len(config)}")
        print(f"targets: {targets}")
        print(f"distractors: {distractors}")
        print(f"ambiguities: {ambiguities}")
        print()
