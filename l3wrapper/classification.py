import subprocess
import os
import pandas as pd
from l3wrapper.dictionary import *
import logging


class L3Classifier:

    BIN_DIR = "bin"
    TRAIN_BIN = "L3CFiltriItemTrain"
    CLASSIFY_BIN = "L3CFiltriItemClassifica"
    CLASSIFICATION_RESULTS = "classificati.txt"
    LEVEL1_FILE = "livelloI.txt"
    LEVEL2_FILE = "livelloII.txt"

    def __init__(self, minsup, minconf, l3_root: str):
        self.minsup = minsup
        self.minconf = minconf
        self.l3_root = l3_root

        bin_dir = os.path.join(l3_root, self.BIN_DIR)
        if not os.path.exists(bin_dir) or not os.path.isdir(bin_dir):
            raise ValueError("bin directory named 'bin' is not present in L3 root")

        self.train_bin_path = os.path.join(self.l3_root, self.BIN_DIR, self.TRAIN_BIN)
        self.classify_bin_path = os.path.join(self.l3_root, self.BIN_DIR, self.CLASSIFY_BIN)

        self.rule_statistics = RuleStatistics()

        self.logger = logging.getLogger(__name__)

    def train_and_predict(self, train: pd.DataFrame,
                          test: pd.DataFrame,
                          columns: list = None,
                          rule_set: str = 'all',  # all, level1, perc, top
                          top_count: int = None,
                          perc_count: int = None,
                          save_train_data_file: bool = False,
                          filtering_rules: dict = None,
                          rule_dictionary: RuleDictionary = None) -> list:
        stem = train.name
        file = '{}.data'.format(stem)

        # training stage
        self.train(train, columns, rule_set, top_count, perc_count, save_train_data_file, filtering_rules,
                   rule_dictionary)

        # classification stage
        if not columns:
            test.to_csv(file, header=False, index=False)
        else:
            test.to_csv(file, header=False, columns=columns, index=False)
        subprocess.run(
            [
                self.classify_bin_path,
                stem,
                self.l3_root
            ],
            stdout=subprocess.DEVNULL
        )

        classification_results = self.read_classification_results()
        return classification_results

    def train(self, train: pd.DataFrame,
              columns: list = None,
              rule_set: str = 'all',  # all, level1, perc, top
              top_count: int = None,
              perc_count: int = None,
              save_train_data_file: bool = False,
              filtering_rules: dict = None,
              rule_dictionary: RuleDictionary = None):

        if rule_set not in ['all', 'level1', 'perc', 'top']:
            raise ValueError("rule_set specified is not valid")
        if rule_set == 'perc' and not perc_count > 0:
            raise ValueError("perc_count must be greater than zero")
        if rule_set == 'top' and not top_count > 0:
            raise ValueError("top_count must be greater than zero")

        stem = train.name
        file = '{}.data'.format(stem)

        if not columns:
            train.to_csv(file, header=False, index=False)
        else:
            train.to_csv(file, header=False, columns=columns, index=False)

        subprocess.run(
            [
                self.train_bin_path,
                stem,
                str(self.minsup), str(self.minconf),
                "nofiltro", "0", "0", "0",
                self.l3_root
            ],
            stdout=subprocess.DEVNULL
        )

        # if rule_set == 'all': do nothing, it's all set
        if rule_set == 'perc':
            # store ${perc_count}% rules and dump them on the same file, then empty level 2
            with open(self.LEVEL1_FILE, 'r') as fp:

                rules_count = sum([1 for _ in fp])
                rules_to_take = int(rules_count * perc_count / 100)

                # enforce at least one rule in the model
                if rules_to_take == 0:
                    rules_to_take = 1

                fp.seek(0, 0)
                top_rules = [fp.readline() for _ in range(rules_to_take)]

            with open(self.LEVEL1_FILE, 'w') as fp:
                # does not add line separators
                fp.writelines(top_rules)

            open(self.LEVEL2_FILE, 'w').close()

        elif rule_set == 'top':
            # store top k rules and dump them on the same file, then empty level 2
            with open(self.LEVEL1_FILE, 'r') as fp:
                top_rules = [fp.readline() for _ in range(top_count)]
                # top_rules = [x for x in fp.readlines()[:top_count]]
            with open(self.LEVEL1_FILE, 'w') as fp:
                fp.writelines(top_rules)

            open(self.LEVEL2_FILE, 'w').close()

        elif rule_set == 'level1':
            open(self.LEVEL2_FILE, 'w').close()     # empty level 2 rules

        if save_train_data_file:  # rename it to not lose it
            os.rename(file, '{}-train.data'.format(stem))

        if filtering_rules:
            class_dict = read_class_dict(stem)                                      # read L3 introduced mappings
            feature_dict = read_feature_dict(stem)
            retained_rules = []

            with open(self.LEVEL1_FILE, 'r') as fp:
                rules_count = 0
                line = fp.readline().strip('\n')

                while line:                                                         # one rule at a time
                    rule = extract_rule(line, 'rule_{}'.format(rules_count + 1), class_dict)
                    self.rule_statistics.rules += [rule]                            # TODO high memory usage!
                    should_discard_rule = False

                    for f_rule_id, f_rule in filtering_rules.items():               # one filter rule at a time
                        filter_triggers = 0

                        for l3_idx in rule.items:
                            attr_idx, attr_val = feature_dict[l3_idx]               # 1-indexed indices
                            if attr_idx > 21:                                       # consider only technical indicators
                                continue
                            if columns:                                             # find the name of the feature
                                feature_name = columns[attr_idx - 1]
                            else:
                                feature_name = train.columns.values[attr_idx - 1]
                            interval_name = rule_dictionary.dict[feature_name][attr_val]
                            if interval_name in f_rule['items'] and rule.label == f_rule['target']:
                                filter_triggers += 1

                        if filter_triggers >= f_rule['sensitivity']:                 # should discard this rule
                            should_discard_rule = True
                            break

                    if not should_discard_rule:
                        retained_rules += ['{}\n'.format(line)]

                    rules_count += 1
                    line = fp.readline().strip('\n')

            with open(self.LEVEL1_FILE, 'w') as fp:
                fp.writelines(retained_rules)

            self.rule_statistics.original_sizes += [rules_count]
            self.rule_statistics.filtered_sizes += [len(retained_rules)]

    def predict(self, data, model_dir: str, columns: list = None) -> list:

        # TODO: check that model_dir contains all the files needed - see README

        stem = os.path.basename(model_dir)
        file = '{}.data'.format(stem)

        # create test file with the same name of the model used in training according to L3 implementation
        if not columns:
            data.to_csv(os.path.join(model_dir, file), header=False, index=False)
        else:
            data.to_csv(os.path.join(model_dir, file), header=False, columns=columns, index=False)
        subprocess.run(
            [
                self.classify_bin_path,
                stem,
                self.l3_root
            ],
            cwd=model_dir,
            stdout=subprocess.DEVNULL
        )

        classification_results = self.read_classification_results(model_dir)
        return classification_results

    def read_classification_results(self, root: str = None) -> list:
        classification = []

        if not root:
            with open(self.CLASSIFICATION_RESULTS, "r") as fp:
                for line in [x.strip() for x in fp.readlines()[2:]]:  # discard first two lines
                    label = line.split(" ")[2]
                    classification += [label]
        else:
            with open(os.path.join(root, self.CLASSIFICATION_RESULTS), "r") as fp:
                for line in [x.strip() for x in fp.readlines()[2:]]:  # discard first two lines
                    label = line.split(" ")[2]
                    classification += [label]

        return classification
