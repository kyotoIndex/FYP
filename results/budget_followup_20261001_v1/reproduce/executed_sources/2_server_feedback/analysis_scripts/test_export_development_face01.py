import pytest
import export_development_face02 as d

def rows():return [{'recipe_id':f'U{i:03d}','seed':s} for i in range(1,33) for s in [503,509,521,541]]
def test_exact_matrix():d.require_matrix(rows())
@pytest.mark.parametrize('kind',['missing','duplicate','unknown_seed'])
def test_matrix_rejects(kind):
 r=rows()
 if kind=='missing':r.pop()
 elif kind=='duplicate':r[-1]=r[0]
 else:r[0]['seed']=601
 with pytest.raises(ValueError):d.require_matrix(r)
