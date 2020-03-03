import pandas as pd
import numpy as np


_UNKNOWN_ITEM = -1


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


class Transaction:
    def __init__(self, row: np.array, item_to_item_id: dict):
        # Extract items in the tuple form (column_id, value) 
        self._items = [(c_id, c_val) for (c_id, c_val) in enumerate(row)]
        
        # Associate to each item its 'item_id'
        self._item_ids = list()
        for item in self._items:
            try:
                self._item_ids.append(item_to_item_id[item])
            except KeyError:
                self._item_ids.append(_UNKNOWN_ITEM)
            
        self.item_ids_set = set(self._item_ids)
        
        self.used_level = None


class Rule:
    def __init__(self, raw_rule: str, rank: int):
        """Extract a new rule from a string (raw rule).

        The ID assigned to the rule is its position in the rules file (either lvl1 or lvl2).
        Also, the Rule stores the list of items as they have been used in classification.
        """
        self.rank = rank
        self.raw_rule = raw_rule

        chunks = raw_rule.split(" ")
        items = chunks[0]

        self.item_ids = set([int(i) for i in items[1:-1].split(",")])
        self._n_items = len(self.item_ids)
        self.class_id = int(chunks[2])
        self.support = int(chunks[3])
        self.confidence = float(chunks[4])

    def get_readable_representation(self, item_id_to_item: dict, column_id_to_name: dict, class_dict: dict) -> str:
        readable_items = []
        for item_id in self.item_ids:
            column_id, value = item_id_to_item[item_id]
            column_name = column_id_to_name[column_id]
            readable_items.append(f"{column_name}:{value}")
        label = class_dict[self.class_id]

        # TODO Should the rule_id be included in the representation?
        return f"{','.join(readable_items)}\t{label}\t{self.support}\t{self.confidence}\t{self._n_items}"

    def match(self, transaction: Transaction) -> bool:
        return self.item_ids.issubset(transaction.item_ids_set)

    def __repr__(self):
        return f"Rule(id:{self.rank};item_ids:{','.join(map(lambda x: str(x), self.item_ids))};sup:{self.support};conf:{self.confidence})"


def build_class_dict(stem: str) -> dict:
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


def build_item_dictionaries(filestem: str) -> (dict, dict):
    """Build two item dictionaries mapping ids created by L3.
    TODO: add description
    TODO: what are 'item_id' and 'item'? 

    :param filestem: name of the dataset used for training
    :return:
    """
    
    item_id_to_item = {}                                            # item_id: (column_id, value)
    with open(f"{filestem}.diz", "r") as fp:
        lines = [l.strip('\n') for l in fp.readlines()]
        for line in lines:
            tok = line.split('>')               # e.g. 74->21,2
            item_id = int(tok[0][:-1])
            attrid_val_pair = tok[1].split(',')
            
            # L3 uses a 1-based positional indexing, hence save 'column_id' as 'attr_pos - 1'
            attr_pos = int(attrid_val_pair[0])
            column_id = attr_pos - 1
            item_id_to_item[item_id] = (column_id, attrid_val_pair[1])
    
    item_to_item_id = {v: k for k, v in item_id_to_item.items()}    # (column_id, value): item_id
    return item_id_to_item, item_to_item_id


def build_columns_dictionary(column_names: list):
    return {c_id: c_name for (c_id, c_name) in enumerate(column_names)}


def parse_raw_rules(filename: str):
    rules = list()
    with open(filename, 'r') as fp:
        rules = [Rule(line.strip('\n'), rank) for (rank, line) in enumerate(fp)]
    return rules


def write_human_readable(filename: str,
                         rules: list,
                         item_id_to_item: dict,
                         column_id_to_name: dict,
                         class_dict: dict):
    with open(filename, 'w') as fp:
        [fp.write(f"{r.get_readable_representation(item_id_to_item, column_id_to_name, class_dict)}\n") \
            for r in rules]
