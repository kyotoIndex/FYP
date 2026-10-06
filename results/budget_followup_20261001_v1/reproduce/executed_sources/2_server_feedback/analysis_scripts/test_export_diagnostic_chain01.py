import copy
import pytest
import export_diagnostic_chain03 as d


def feedback():
    corners={k:{'public_NLL':v,'independent_NLL':v, 'state_hash':k,
                'alpha':int(k[0]),'beta':int(k[1])}
             for k,v in [('00',1.),('10',.9),('01',.8),('11',.6)]}
    rules={k:{'alpha':1.,'beta':1.,'selected_state_hash':'11','public_NLL':.6,
        'independent_NLL':.6,'public_NLL_source':'own_standalone_public_grid',
        'public_gain_vs_parent':.4,'independent_gain_vs_parent':.4,
        'public_independent_gain_direction_agrees':True,
        'standalone_measured_server_wall_seconds':.5} for k in d.RULES}
    clients=[{'client_id':i,'sample_weight':.1,'child_state_hash':str(i),
        'local_parent':{'loss':1.,'samples':64},'local_child':{'loss':.9,'samples':64},
        'individual_global':{'loss':.8,'samples':2048},'local_NLL_gain':.1,
        'individual_global_NLL_gain':.2,'weighted_individual_global_NLL_gain':.02}
        for i in range(10)]
    return {'selection_role':'public_training_512','independent_readout_role':'cohort_readout_2048',
        'selection_completed_before_independent_readout':True,'rules':rules,'four_corners':corners,
        'independent_readout_by_state_hash':{'11':{'loss':.6,'accuracy':.8,'samples':2048}},
        'transfer_chain':{'clients':clients,'global_parent':{'loss':1.},
            'aggregate':{'loss':.6},'weighted_individual_global_NLL_gain':.2,
            'aggregate_global_NLL_gain':.4},
        'interaction_I':{'public':.1,'independent':.1},
        'posthoc_readout_grid':{'best_independent_NLL':.5,
            'role':'diagnostic_only_after_public_selection'}}


def test_chain_weight_applied_once_and_aggregate_separate():
    rules,clients,corners,chain=d.project_feedback({'seed':101},'local',feedback())
    assert len(rules)==5 and len(clients)==10 and len(corners)==4
    assert chain['weighted_individual_global_NLL_gain']==pytest.approx(.2)
    assert chain['aggregate_global_NLL_gain']==pytest.approx(.4)
    assert chain['aggregate_minus_sum_individual']==pytest.approx(.2)
    assert rules[0]['independent_gain_fraction_parent']==pytest.approx(.4)


@pytest.mark.parametrize('failure',['double_weight','duplicate_client','selection_order','readout_count','interaction','selected_loss'])
def test_wrong_chain_rejected(failure):
    f=feedback()
    if failure=='double_weight':f['transfer_chain']['clients'][0]['weighted_individual_global_NLL_gain']=.002
    elif failure=='duplicate_client':f['transfer_chain']['clients'][0]['client_id']=1
    elif failure=='selection_order':f['selection_completed_before_independent_readout']=False
    elif failure=='readout_count':f['independent_readout_by_state_hash']['11']['samples']=1024
    elif failure=='interaction':f['interaction_I']['independent']=.2
    else:f['rules']['block16']['independent_NLL']=.7
    with pytest.raises(ValueError):d.project_feedback({'seed':101},'local',f)


def test_exact_prescribed_object_matrix():
    jobs=d.expected_jobs()
    assert len(jobs)==120
    assert sum(j.startswith('BANK_') for j in jobs)==96
    assert sum(j.startswith('R8_TRANSFER_') for j in jobs)==24


def test_actual_frozen_support_contract():
    assert d.SUPPORTS == {'BANK_READOUT':['local','cyclic'], 'R8_TRANSFER':['rank8']}


def test_null_reference_remains_null_and_raw_excluded():
    assert d.old_reference_provenance(None) is None
    assert d.old_reference_provenance({"result_sha256":"abc","raw":{"logits":[1]}})=={"result_sha256":"abc"}
    with pytest.raises(ValueError): d.old_reference_provenance([])
