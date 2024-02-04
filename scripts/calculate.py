import sys
import os
import glob
from tabulate import tabulate


def main():
    dirpath = sys.argv[1]
    file_pattern = sys.argv[2]

    results = {
        'Gradient Ascent only': {
            "Mean Fitness": 0,
            "Orc. Mean Fitness": 0,
            "Max Fitness": 0,
            "Orc. Max Fitness": 0,
            "Diversity": 0,
            "dist(WT)": 0
        },
        'Gradient Ascent + Directed Evolution': {
            "Mean Fitness": 0,
            "Orc. Mean Fitness": 0,
            "Max Fitness": 0,
            "Orc. Max Fitness": 0,
            "Diversity": 0,
            "dist(WT)": 0
        },
        'Directed Evolution only': {
            "Mean Fitness": 0,
            "Orc. Mean Fitness": 0,
            "Max Fitness": 0,
            "Orc. Max Fitness": 0,
            "Diversity": 0,
            "dist(WT)": 0
        }
    }

    filepaths = glob.glob(os.path.join(dirpath, f"{file_pattern}_*.txt"))
    counter = 0
    for path in filepaths:
        contents = open(path, "r").readlines()
        for i in [4, 11, 18]:
            content = contents[i].split()
            if len(content) == 7:
                if counter == 0:
                    key = "Gradient Ascent only"
                elif counter == 1:
                    key = "Gradient Ascent + Directed Evolution"
                elif counter == 2:
                    key = "Directed Evolution only"

                results[key]["Mean Fitness"] += float(content[1])
                results[key]['Orc. Mean Fitness'] += float(content[2])
                results[key]['Max Fitness'] += float(content[3])
                results[key]['Orc. Max Fitness'] += float(content[4])
                results[key]['Diversity'] += float(content[5])
                results[key]['dist(WT)'] += float(content[6])

                counter += 1
        counter = 0

    exp_types = list(results.keys())

    saved_file = os.path.join(dirpath, f"{file_pattern}_total.txt")
    contents = []
    for exp in exp_types:
        # print(exp)
        contents.append(exp)
        table = tabulate(
            [[
                results[exp]['Mean Fitness'] / len(filepaths),
                results[exp]['Orc. Mean Fitness'] / len(filepaths),
                results[exp]['Max Fitness'] / len(filepaths),
                results[exp]['Orc. Max Fitness'] / len(filepaths),
                results[exp]['Diversity'] / len(filepaths),
                results[exp]['dist(WT)'] / len(filepaths)
            ]],
            headers=["Mean fitness", "Orc. Mean fitness", "Max fitness",
                     "Orc. Max fitness", "Diversity", "dist(WT)"]
        )
        print(table)
        print("*" * 10)
        contents.extend([table, "*" * 10])
        with open(saved_file, "w") as f:
            f.write("\n".join(contents))


if __name__ == "__main__":
    main()
