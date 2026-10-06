import json
import pytest
from pathlib import Path
from export_execution_cost01 import measured_timing

def test_disjoint_categories_and_partial_retained(tmp_path):
 p=tmp_path/'timing.json';p.write_text(json.dumps({'schema':'pcla-exclusive-host-timing-2','category_seconds':{'necessary_algorithm':2,'research_observation':1},'total_seconds':3,'partial':True,'tail_unmeasured':True}));v=measured_timing(p);assert v['partial'] and v['tail_unmeasured']
def test_overlapping_or_corrupt_total_rejected(tmp_path):
 p=tmp_path/'timing.json';p.write_text(json.dumps({'schema':'pcla-exclusive-host-timing-2','category_seconds':{'necessary_algorithm':2,'research_observation':1},'total_seconds':2}))
 with pytest.raises(ValueError):measured_timing(p)
