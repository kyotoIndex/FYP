import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('v7_dispatch', Path(__file__).resolve().parents[1]/'scripts/dispatch_pcla_v7.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'model.py').write_text('version = 1\n')
        d.write(self.source/'SOURCE_MANIFEST.json', {'files': {'model.py':d.sha(self.source/'model.py')}})
        self.configs = self.root/'configs'
        self.configs.mkdir()
        d.write(self.configs/'j.json', {'model': 'test'})
        self.queue = self.root/'queue'
        self.path = self.queue/'jobs/j.json'
        d.write(self.path, {'job': {'job_id': 'j', 'seed':401, 'priority':20, 'hardware_block':'same-seed-lr'},
                           'pool':'4090D', 'state':'blocked', 'attempts':[]})

    def release(self):
        d.release(self.queue, ['j'], self.configs, self.source, self.root/'out')

    def test_claim_is_single_owner_and_hardware_bound(self):
        self.release()
        self.assertIsNone(d.claim(self.queue,'A100','wrong-model'))
        p, v = d.claim(self.queue,'4090D','first')
        self.assertEqual(v['worker'],'first')
        self.assertIsNone(d.claim(self.queue,'4090D','second'))
        with self.assertRaises(RuntimeError):
            d.finish(self.queue,p,'second','executed_pending_audit')
        d.finish(self.queue,p,'first','executed_pending_audit')
        self.assertEqual(d.read(p)['state'],'executed_pending_audit')

    def test_mutated_config_is_rejected_before_run(self):
        self.release()
        d.write(self.configs/'j.json', {'model':'different'})
        with self.assertRaises(RuntimeError): d.claim(self.queue,'4090D','w')
        self.assertEqual(d.read(self.path)['state'],'pending')

    def test_cost_seed_requires_exclusive_gpu(self):
        self.release()
        self.assertIsNone(d.claim(self.queue, '4090D', 'shared', mode='shared'))
        p, v = d.claim(self.queue, '4090D', 'exclusive', mode='exclusive')
        self.assertEqual(v['worker'], 'exclusive')

    def test_mutated_source_is_rejected_before_run(self):
        self.release()
        (self.source/'model.py').write_text('version = 2\n')
        with self.assertRaises(RuntimeError): d.claim(self.queue,'4090D','w')
        self.assertEqual(d.read(self.path)['state'],'pending')

    def test_release_does_not_overwrite_running_work(self):
        self.release()
        d.claim(self.queue,'4090D','w')
        with self.assertRaises(ValueError): self.release()
        self.assertEqual(d.read(self.path)['state'],'running')


if __name__ == '__main__': unittest.main()
