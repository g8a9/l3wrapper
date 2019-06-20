import pandas as pd


class RuleDictionary:
    """
    Handler class providing a way to store textual representation of association rule items. Implementation supports
    discrete attributes belonging to two or three bins.
    """

    def __init__(self):
        self.dict = {}

    def add_attribute(self, name: str, tr1: int, tr2: int = None):
        double = False
        if tr2:
            if tr2 <= tr1:
                raise ValueError("Threshold 2 value is not correct")
            else:
                double = True

        if double:
            self.dict[name] = {}
            self.dict[name]["1"] = '{}<={}'.format(name, tr1)
            self.dict[name]["2"] = '{}<{}<{}'.format(tr1, name, tr2)
            self.dict[name]["3"] = '{}>={}'.format(name, tr2)
        else:
            self.dict[name] = {}
            self.dict[name]["1"] = '{}<{}'.format(name, tr1)
            self.dict[name]["2"] = '{}>={}'.format(name, tr1)


class Rule:
    def __init__(self,
                 name: str,
                 label: str,
                 support: int,
                 confidence: float,
                 items: [],
                 raw: str):
        self.name = name
        self.label = label
        self.support = support
        self.confidence = confidence
        self.items = items
        self.raw = raw

    def to_string(self, rules_list, rules_dictionary: RuleDictionary):
        readable_items = []
        for item in self.items:
            index = int(item) - 1                       # comment this line if items numbering is 0-indexed
            attribute, value = rules_list[index]
            readable_items += [rules_dictionary.dict[attribute][value]]
        return '{}, {} -> {}, {}, {}'.format(self.name, readable_items, self.label, self.support, self.confidence)

    def to_dict(self, rules_list, rules_dictionary: RuleDictionary):
        d = {
            "Name": self.name,
            "Support": self.support,
            "Confidence": self.confidence,
            "Label": self.label,
            "Items": []
        }
        for item in self.items:
            index = int(item) - 1                       # comment this line if items numbering is 0-indexed
            attribute, value = rules_list[index]
            d["Items"] += [rules_dictionary[attribute][value]]
        return d


class RuleStatistics:
    def __init__(self):
        self.rules = []
        self.original_sizes = []
        self.filtered_sizes = []

    def print_statistics(self):
        original = pd.Series(self.original_sizes)
        filtered = pd.Series(self.filtered_sizes)
        perc_filtered = 100 * ((filtered - original) / original)
        print('Average {}% removed considering all set of rules filtered'.format(perc_filtered.mean()))


def read_class_dict(stem: str) -> dict:
    """
    Read numerical label id assigned by L3 implementation to class labels
    :param stem: name of the dataset used for training
    :return: class id dictionary
    """
    class_dict = {}
    with open('{}.cls'.format(stem), 'r') as fp:
        lines = [f.strip('\n') for f in fp.readlines()]
    label_id = int(lines[0])
    for label in lines[1:]:
        class_dict[label_id] = label
        label_id += 1
    return class_dict


def read_feature_dict(stem: str) -> dict:
    """
    Read numerical id assigned by L3 implementation to each pair (attribute_index,value).
    Note: attribute index is got enumerating of attributes, starting from 1.
    :param stem: name of the dataset used for training
    :return: class id dictionary
    """
    feature_dict = {}
    with open('{}.diz'.format(stem), 'r') as fp:
        lines = [l.strip('\n') for l in fp.readlines()]
        for line in lines:
            tok = line.split('>')               # e.g. 74->21,2
            l3_idx = int(tok[0][:-1])
            l3_val = tok[1].split(',')
            feature_dict[l3_idx] = (int(l3_val[0]), l3_val[1])
    return feature_dict


def extract_rule(raw_rule: str, name: str, class_dict: {}) -> Rule:
    """
    Extract a new rule from a strings. The name/ID assigned is their position in the list.
    """
    chunks = raw_rule.split(" ")
    items = chunks[0]
    items = [int(i) for i in items[1:-1].split(",")]
    raw_label = int(chunks[2])
    label = class_dict[raw_label]
    support = int(chunks[3])
    confidence = float(chunks[4])

    return Rule(name, label, support, confidence, items, raw_rule)
