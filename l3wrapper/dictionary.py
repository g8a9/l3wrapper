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
                 rule_id: str,
                 label: str,
                 support: int,
                 confidence: float,
                 items: [],
                 feature_names: list,
                 raw: str):
        self.rule_id = rule_id
        self.label = label
        self.support = support
        self.confidence = confidence
        self.items = items
        self.feature_names = feature_names
        self.raw = raw

    def to_string(self, item_dict: dict, rule_dict: RuleDictionary):
        readable_items = []
        for item_id in self.items:
            feature_id, value = item_dict[item_id]
            feature_name = self.feature_names[feature_id - 1]
            item_as_string = (rule_dict.dict[feature_name])[value]
            readable_items.append(item_as_string)
        return '{},{},{},{},{}'.format(self.rule_id, ','.join(readable_items), self.label, self.support, self.confidence)

    # def to_dict(self, rules_list, rules_dictionary: RuleDictionary):
    #     d = {
    #         "Name": self.name,
    #         "Support": self.support,
    #         "Confidence": self.confidence,
    #         "Label": self.label,
    #         "Items": []
    #     }
    #     for item in self.items:
    #         index = int(item) - 1                       # comment this line if items numbering is 0-indexed
    #         attribute, value = rules_list[index]
    #         d["Items"] += [rules_dictionary[attribute][value]]
    #     return d


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


def read_item_dict(stem: str) -> dict:
    """
    Read numerical id assigned by L3 implementation to each pair (attribute_index,value).
    Note: attribute index is obtained enumerating of attributes, starting from 1.
    :param stem: name of the dataset used for training
    :return: rule's item dictionary
    """
    item_dict = {}
    with open('{}.diz'.format(stem), 'r') as fp:
        lines = [l.strip('\n') for l in fp.readlines()]
        for line in lines:
            tok = line.split('>')               # e.g. 74->21,2
            item_id = int(tok[0][:-1])
            attrid_val_pair = tok[1].split(',')
            item_dict[item_id] = (int(attrid_val_pair[0]), attrid_val_pair[1])
    return item_dict


def extract_rule(raw_rule: str, rule_id: str, class_dict: {}, feature_names: list) -> Rule:
    """Extract a new rule from a string (raw rule).

    The ID assigned to the rule is its position in the list. Also, the Rule stores the list
    of features as they have been used in classification. This is necessary since L3 implementation
    assigns to each item of the rule a positive integer id (starting from 1) representing a positional
    index of the feature.
    :return: the formatted rule (see Rule class)
    """
    chunks = raw_rule.split(" ")
    items = chunks[0]
    items = [int(i) for i in items[1:-1].split(",")]
    raw_label = int(chunks[2])
    label = class_dict[raw_label]
    support = int(chunks[3])
    confidence = float(chunks[4])
    return Rule(rule_id, label, support, confidence, items, feature_names, raw_rule)
