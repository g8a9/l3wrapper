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
    def __init__(self, raw_rule: str, rule_id: str):
        """Extract a new rule from a string (raw rule).

        The ID assigned to the rule is its position in the rules file (either lvl1 or lvl2).
        Also, the Rule stores the list of items as they have been used in classification.
        """
        self.rule_id = rule_id
        self.raw_rule = raw_rule

        chunks = raw_rule.split(" ")
        items = chunks[0]
        self.items = [int(i) for i in items[1:-1].split(",")]
        self._n_items = len(self.items)
        self.label = int(chunks[2])
        self.support = int(chunks[3])
        self.confidence = float(chunks[4])

    def get_readable_representation(self, item_dict: dict, class_dict: dict):
        readable_items = []
        for item_id in self.items:
            column_name, value = item_dict[item_id]
            readable_items.append(f"{column_name}:{value}")
        label = class_dict[self.label]

        # TODO Should the rule_id be included in the representation?
        return f"{','.join(readable_items)}\t{label}\t{self.support}\t{self.confidence}\t{self._n_items}"


def read_class_dict(stem: str) -> dict:
    """
    Read numerical label id assigned by L3 implementation to class labels
    :param stem: name of the dataset used for training
    :return: class id dictionary
    """
    class_dict = {}
    with open(f"{stem}.cls", "r") as fp:
        lines = [f.strip('\n') for f in fp.readlines()]
    label_id = int(lines[0])
    for label in lines[1:]:
        class_dict[label_id] = label
        label_id += 1
    return class_dict


def read_item_dict(stem: str, column_names: list) -> dict:
    """
    Read numerical id assigned by L3 implementation to each pair (attribute_index,value).
    Note: attribute index is obtained enumerating of attributes, starting from 1.
    :param stem: name of the dataset used for training
    :return: rule's item dictionary
    """
    item_dict = {}
    with open(f"{stem}.diz", "r") as fp:
        lines = [l.strip('\n') for l in fp.readlines()]
        for line in lines:
            tok = line.split('>')               # e.g. 74->21,2
            item_id = int(tok[0][:-1])
            attrid_val_pair = tok[1].split(',')
            
            attr_pos = int(attrid_val_pair[0])  # this is 1-based positional indexing
            item_dict[item_id] = (column_names[attr_pos - 1], attrid_val_pair[1])
    return item_dict


def parse_raw_rules(filename: str):
    rules = list()
    with open(filename, 'r') as fp:
        rules = [Rule(line.strip('\n'), f"rule_{i}") for (i, line) in enumerate(fp)]
    return rules


def write_human_readable(filename: str, rules: list, item_dict: dict, class_dict: dict):
    with open(filename, 'w') as fp:
        [fp.write(f"{r.get_readable_representation(item_dict, class_dict)}\n") for r in rules]
