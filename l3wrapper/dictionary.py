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
        self.used_level = -1
        self.matched_rules = None


class Rule:
    def __init__(self, raw_rule: str, rule_id: int):
        """Extract a new rule from a string (raw rule).

        The Rule stores the list of items as they have been used in classification.

        Parameters
        ----------
        raw_rule : str
            A plain string as extracted by the L3 training.
        rule_id : int
            An integer corresponding to the rule id.
        """
        self.rule_id = rule_id
        self.raw_rule = raw_rule

        chunks = raw_rule.split(" ")
        items = chunks[0]

        self.item_ids = set([int(i) for i in items[1:-1].split(",")])
        self._n_items = len(self.item_ids)
        self.class_id = int(chunks[2])
        self.support = int(chunks[3])
        self.confidence = float(chunks[4])

    def get_readable_representation(self, item_id_to_item: dict, column_id_to_name: dict, class_dict: dict) -> str:
        """Create a human readable representation of the rule.

        Before getting to the final representation, the rule's items are sorted
        by their respective column_id. As such, the represented itemset will begin
        with items related to the first column, then to the second one, and so on.
        """
        readable_items = list()
        sorted_items = sorted(
            list(self.item_ids), key=lambda i: item_id_to_item[i][0]
        )
        for item_id in sorted_items:
            column_id, value = item_id_to_item[item_id]
            column_name = column_id_to_name[column_id]
            readable_items.append(f"{column_name}:{value}")
        label = class_dict[self.class_id]

        return (
            f"{self.rule_id}\t"
            f"{','.join(readable_items)}\t"
            f"{label}\t"
            f"{self.support}\t"
            f"{self.confidence}\t"
            f"{self._n_items}"
        )

    def match(self, transaction: Transaction) -> bool:
        return self.item_ids.issubset(transaction.item_ids_set)

    def __repr__(self):
        return f"Rule(id:{self.rule_id};item_ids:{','.join(map(lambda x: str(x), self.item_ids))};sup:{self.support};conf:{self.confidence})"


def build_y_mappings(y: np.array) -> dict:
    orig_to_str = {label: str(label) for label in y}
    str_to_orig = {v: k for k, v in orig_to_str.items()}
    return orig_to_str, str_to_orig


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
    """Build two item dictionaries mapping the ids created by the L3 C binaries.

    Parameters
    ----------
    filestem : str
        The set of the training file. The methods will look for a file named <filestem>.diz

    Returns
    -------
    item_id_to_item, item_to_item_id : (dict, dict)
        A tuple containing as the first element a dictionary mapping from item
        id to the tuple (column_id,value), and as second element a dictionary
        with the inverse mapping.
    """

    # item_id: (column_id, value)
    item_id_to_item = {}
    with open(f"{filestem}.diz", "r") as fp:
        lines = [l.strip('\n') for l in fp.readlines()]
        for line in lines:
            tok = line.split('->')               # e.g. 74->21,2
            item_id = int(tok[0])
            right_end = "->".join(tok[1:])
            attrid_val_pair = right_end.split(',')

            # L3 uses a 1-based positional indexing, hence save 'column_id' as 'attr_pos - 1'
            attr_pos = int(attrid_val_pair[0])
            column_id = attr_pos - 1
            item_id_to_item[item_id] = (column_id, attrid_val_pair[1])

    # (column_id, value): item_id
    item_to_item_id = {v: k for k, v in item_id_to_item.items()}
    return item_id_to_item, item_to_item_id


def build_columns_dictionary(column_names: list):
    return {c_id: c_name for (c_id, c_name) in enumerate(column_names)}


def parse_raw_rules(filename: str):
    """Given a file with raw rules, parse and extract a list of Rule.

    The ID assigned to the rule is its position in the rules file (either lvl1 or lvl2).

    Parameters
    ----------
    filename : str
        Name of the file where the raw rules are saved.

    Returns
    -------
    rules : list
        A list of Rule extracted from <filename>.
    """
    rules = list()
    with open(filename, 'r') as fp:
        rules = [Rule(line.strip('\n'), rule_id) for (rule_id, line) in enumerate(fp)]
    return rules


def write_human_readable(filename: str,
                         rules: list,
                         item_id_to_item: dict,
                         column_id_to_name: dict,
                         class_dict: dict):
    with open(filename, 'w') as fp:
        [fp.write(f"{r.get_readable_representation(item_id_to_item, column_id_to_name, class_dict)}\n")
            for r in rules]
