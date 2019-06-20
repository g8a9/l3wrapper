from l3wrapper.dictionary import *


stem = "A2A.MI"


def test_read_class_dict():
    class_dict = read_class_dict(stem)
    assert len(class_dict) == 3
    for lab, tr in zip(class_dict.keys(), [2147483548, 2147483549, 2147483550]):
        assert lab == tr


def test_read_feature_dict():
    feature_dict = read_feature_dict(stem)
    assert len(feature_dict) == 74
    for lab, tr in zip(feature_dict.keys(), range(1, 75)):
        assert lab == tr
