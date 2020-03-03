import subprocess
import os
import pandas as pd
from l3wrapper.dictionary import *
import logging
import warning

FILTER_LVL1 = 1
FILTER_LVL2 = 2
FILTER_LVL12 = 12

raise DeprecationWarning("The usage of this module is deprecated.")


class L3Classifier:

    BIN_DIR = "bin"
    TRAIN_BIN = "L3CFiltriItemTrain"
    CLASSIFY_BIN = "L3CFiltriItemClassifica"
    CLASSIFICATION_RESULTS = "classificati.txt"
    LEVEL1_FILE = "livelloI.txt"
    LEVEL2_FILE = "livelloII.txt"
    LEVEL1_FILE_READABLE = "livelloI_readable.txt"
    LEVEL2_FILE_READABLE = "livelloII_readable.txt"
    FILTER_LEVEL1 = ''
    FILTER_LEVEL2 = ''
    FILTER_BOTH = ''

    def __init__(self, minsup, minconf, l3_root: str):
        self.minsup = minsup
        self.minconf = minconf
        self.l3_root = l3_root

        bin_dir = os.path.join(l3_root, self.BIN_DIR)
        if not os.path.exists(bin_dir) or not os.path.isdir(bin_dir):
            raise ValueError("bin directory named 'bin' is not present in L3 root")

        self.train_bin_path = os.path.join(self.l3_root, self.BIN_DIR, self.TRAIN_BIN)
        self.classify_bin_path = os.path.join(self.l3_root, self.BIN_DIR, self.CLASSIFY_BIN)
        self.logger = logging.getLogger(__name__)

    def train_and_predict(self, train: pd.DataFrame,
                          test: pd.DataFrame,
                          columns: list = None,
                          rule_set: str = 'all',  # all, level1, perc, top
                          top_count: int = None,
                          perc_count: int = None,
                          save_train_data_file: bool = False,
                          filter_level: str = None,
                          filtering_rules: dict = None,
                          rule_dictionary: RuleDictionary = None) -> list:
        stem = train.name
        file = '{}.data'.format(stem)

        # training stage
        self.train(train, columns=columns, rule_set=rule_set, top_count=top_count, perc_count=perc_count,
                   save_train_data_file=save_train_data_file,
                   filtering_rules=filtering_rules,
                   filter_level=filter_level, rule_dictionary=rule_dictionary)

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
              save_human_readable: bool = True,
              filter_level: str = None,
              filtering_rules: dict = None,
              rule_dictionary: RuleDictionary = None,
              save_rules_readable_format: bool = False):

        if rule_set not in ['all', 'level1', 'perc', 'top']:
            raise ValueError("rule_set specified is not valid")
        if rule_set == 'perc' and not perc_count > 0:
            raise ValueError("perc_count must be greater than zero")
        if rule_set == 'top' and not top_count > 0:
            raise ValueError("top_count must be greater than zero")
        if save_rules_readable_format and rule_dictionary is None:
            raise ValueError("saving rules in readable format without a rule dictionary")

        stem = train.name
        file = '{}.data'.format(stem)

        if not columns:
            train.to_csv(file, header=False, index=False)
            feature_names = train.columns.values
        else:
            train.to_csv(file, header=False, columns=columns, index=False)
            feature_names = columns

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

        class_dict = read_class_dict(stem)  # read L3 introduced mappings
        item_dict = read_item_dict(stem)
        lvl1_raw_r, lvl1_r = [], []
        lvl2_raw_r, lvl2_r = [], []

        if not filtering_rules:
            raise RuntimeError('Filtering rules are not filled')

        # filter rules beforehand, do not save the original ones
        if filter_level == FILTER_LVL1:
            lvl1_raw_r, lvl1_r = self.filter_level_rules(self.LEVEL1_FILE, rule_dictionary, filtering_rules, class_dict,
                                                         item_dict, train, feature_names)
            with open(self.LEVEL1_FILE, 'w') as fp:
                    [fp.write(r) for r in lvl1_raw_r]

        elif filter_level == FILTER_LVL2:
            lvl2_raw_r, lvl2_r = self.filter_level_rules(self.LEVEL2_FILE, rule_dictionary, filtering_rules, class_dict,
                                                         item_dict, train, feature_names)
            with open(self.LEVEL2_FILE, 'w') as fp:
                    [fp.write(r) for r in lvl2_raw_r]

        elif filter_level == FILTER_LVL12:
            lvl1_raw_r, lvl1_r = self.filter_level_rules(self.LEVEL1_FILE, rule_dictionary, filtering_rules, class_dict,
                                                         item_dict, train, feature_names)
            lvl2_raw_r, lvl2_r = self.filter_level_rules(self.LEVEL2_FILE, rule_dictionary, filtering_rules, class_dict,
                                                         item_dict, train, feature_names)
            with open(self.LEVEL1_FILE, 'w') as fp:
                    [fp.write(r) for r in lvl1_raw_r]
            with open(self.LEVEL2_FILE, 'w') as fp:
                    [fp.write(r) for r in lvl2_raw_r]

        # translate the model to human readable format
        if save_human_readable:
            # here the filter procedure has already parsed the rule to be retained
            if filter_level:
                with open(self.LEVEL1_FILE_READABLE, 'w') as fp:
                    [fp.write('{}\n'.format(r.to_string(item_dict, rule_dictionary))) for r in lvl1_r]
                with open(self.LEVEL2_FILE_READABLE, 'w') as fp:
                    [fp.write('{}\n'.format(r.to_string(item_dict, rule_dictionary))) for r in lvl2_r]
            else:
                # parse all the rules and save them
                with open(self.LEVEL1_FILE, 'r') as fp:
                    for i, line in enumerate(fp):
                        lvl1_r.append(extract_rule(line.strip('\n'), 'rule_{}'.format(i), class_dict, feature_names))
                with open(self.LEVEL2_FILE, 'r') as fp:
                    for i, line in enumerate(fp):
                        lvl2_r.append(extract_rule(line.strip('\n'), 'rule_{}'.format(i), class_dict, feature_names))
                with open(self.LEVEL1_FILE_READABLE, 'w') as fp:
                    [fp.write('{}\n'.format(r.to_string(item_dict, rule_dictionary))) for r in lvl1_r]
                with open(self.LEVEL2_FILE_READABLE, 'w') as fp:
                    [fp.write('{}\n'.format(r.to_string(item_dict, rule_dictionary))) for r in lvl2_r]


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

    def filter_level_rules(self, level_file: str, rule_dictionary: RuleDictionary,
                           filtering_rules: dict, class_dict: dict, item_dict: dict,
                           train: pd.DataFrame, feature_names):
        retained_rules_raw = []
        retained_rules_readable = []

        with open(level_file, 'r') as fp:
            for i, line in enumerate(fp):
                rule = extract_rule(line.strip('\n'), 'rule_{}'.format(i), class_dict, feature_names)
                should_discard_rule = False

                for f_rule_id, f_rule in filtering_rules.items():
                    filter_triggers = 0

                    for l3_idx in rule.items:
                        attr_idx, attr_val = item_dict[l3_idx]  # 1-indexed indices
                        # if attr_idx > 21:  # consider only technical indicators
                        #     continue
                        feature_name = feature_names[attr_idx - 1]
                        interval_name = rule_dictionary.dict[feature_name][attr_val]
                        if interval_name in f_rule['items'] and rule.label == f_rule['target']:
                            filter_triggers += 1

                    if filter_triggers >= f_rule['sensitivity']:  # should discard this rule
                        should_discard_rule = True
                        break

                if not should_discard_rule:
                    retained_rules_raw.append(line)
                    retained_rules_readable.append(rule)

        return retained_rules_raw, retained_rules_readable
