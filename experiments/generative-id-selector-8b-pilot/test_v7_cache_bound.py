from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
import ci_common, run_ci_pair, run_id_selector_case, run_smoke_cases
ROOT=Path(__file__).resolve().parent
class V7CacheBoundTests(unittest.TestCase):
    def test_manifest_pin_and_runtime_config_are_coherent(self):
        manifest=ci_common.reviewed_artifact_manifest()
        self.assertEqual(manifest['server']['cache_ram_mib'],0)
        self.assertEqual(run_ci_pair.SERVER_CONFIG['cache_ram_mib'],0)
        self.assertEqual(run_ci_pair.SERVER_CONFIG,manifest['server'])
        self.assertEqual(run_ci_pair.ARTIFACT_SHA256,ci_common.ARTIFACT_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256((ROOT/'GENERATION_DATASET_MANIFEST.json').read_bytes()).hexdigest(),ci_common.GENERATION_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256((ROOT/'ARTIFACT_MANIFEST.json').read_bytes()).hexdigest(),ci_common.ARTIFACT_MANIFEST_SHA256)
    def test_command_adds_only_explicit_zero_cache_bound(self):
        command=run_ci_pair.build_server_command(Path('/runtime/llama-server'),Path('/model/qwen.gguf'))
        expected=[str(Path('/runtime/llama-server')),'--model',str(Path('/model/qwen.gguf')),'--host','127.0.0.1','--alias','Qwen3-8B-Q4_K_M.gguf','--port','50193','--ctx-size','6144','--threads','4','--threads-batch','4','--batch-size','128','--ubatch-size','128','--parallel','1','--n-gpu-layers','0','--reasoning','off','--n-predict','256','--seed','20261023','--jinja','--no-context-shift','--cache-ram','0']
        self.assertEqual(command,expected)
        self.assertEqual(command.count('--cache-ram'),1)
    def test_selector_provider_is_pinned_at_all_entrypoints(self):
        digest=hashlib.sha256((ROOT/'generative_provider.py').read_bytes()).hexdigest()
        self.assertEqual(digest,'48d41efaf6aee247e30da0d6a3f86988003190813ecfcce8fd1743ab5d558149')
        self.assertEqual(run_ci_pair.PROVIDER_SHA256,digest)
        self.assertEqual(run_id_selector_case.PROVIDER_SHA256,digest)
        self.assertEqual(run_smoke_cases.PROVIDER_SHA256,digest)
    def test_other_frozen_configs_are_unchanged(self):
        manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['generation'],{'seed':20261023,'temperature':0.0,'enable_thinking':False,'max_tokens':256,'context_tokens':6144,'max_input_tokens':5632})
        self.assertEqual(manifest['resource_scope']['memory_max_bytes'],10737418240)
        self.assertEqual(manifest['resource_scope']['memory_swap_max_bytes'],0)
        self.assertEqual(manifest['resource_scope']['cpu_quota'],'400%')
        self.assertEqual(manifest['server']['parallel'],1)
        self.assertEqual(manifest['server']['batch_size'],128)
        self.assertEqual(manifest['server']['ubatch_size'],128)
if __name__=='__main__': unittest.main()
