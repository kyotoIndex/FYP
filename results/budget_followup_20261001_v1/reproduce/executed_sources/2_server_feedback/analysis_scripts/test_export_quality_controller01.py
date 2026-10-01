import math
import pytest
from export_quality_controller02 import classification_stats

def test_balanced_accuracy_respects_truth_class_denominators():
    s=classification_stats([[1,0],[0,1],[0,2],[0,1]],[0,0,1,1])
    assert s['accuracy']==0.75 and s['balanced_accuracy']==0.75
    assert s['prediction_class_0']==1 and s['prediction_class_1']==3
    assert s['correct_class_0']==1 and s['correct_class_1']==2

def test_imbalanced_accuracy_and_balanced_are_distinct():
    s=classification_stats([[2,0]]*4,[0,0,0,1])
    assert s['accuracy']==0.75 and s['balanced_accuracy']==0.5

def test_ties_first_class_and_stable_extreme_logits():
    s=classification_stats([[0,0],[1000,-1000],[-1000,1000]],[0,0,1])
    assert s['accuracy']==1 and s['NLL']==pytest.approx(math.log(2)/3)
    assert s['prediction_class_0']==2

def test_absent_class_balanced_is_null_with_reason():
    s=classification_stats([[0,1],[0,1]],[1,1])
    assert s['balanced_accuracy'] is None and s['balanced_accuracy_missing_reason']

@pytest.mark.parametrize('logits,labels',[([],[]),([[0,1]],[0,1]),([[0,float('nan')]],[0]),([[1]],[0]),([[0,1]],[2])])
def test_reject_bad_domain_and_coverage(logits,labels):
    with pytest.raises(ValueError):classification_stats(logits,labels)
