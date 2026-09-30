"""Control-flow and output-policy tests without loading speech models."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import onnx_pipeline as p
import onnx_runtime
import onnx_staged_runtime

class PipelineTests(unittest.TestCase):
    def test_base_export_rejects_unapplied_voice_adaptations(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile=Path(tmp)/'voice.json'
            for setting in [{'adapter':'candidate.pt'}, {'mel_calibration':'candidate.json'}]:
                profile.write_text(json.dumps(setting))
                with self.assertRaisesRegex(ValueError,'Use nano-clone'):
                    p._load_profile(path=profile)
            profile.write_text('{}')
            self.assertEqual(p._load_profile(path=profile)[1],{})

    def invoke_tokens(self, sequence, limit=700):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'tokens').mkdir(); (root/'prepare').mkdir()
            (root/'tokens/config.json').write_text(json.dumps({'mode':'t3_pending'}))
            (root/'prepare/profile.json').write_text('{}')
            class Runtime:
                def __init__(self,*a,**kw): self.step=1
                def outputs(self):
                    return np.array([[self.step]],dtype=np.float32), [np.zeros((1,1,self.step,1))]
                def prefill(self,*a): return self.outputs()
                def decode(self,*a): self.step+=1; return self.outputs()
            observed=[]
            def sample(logits,history,**kwargs):
                observed.append((float(logits[0]),list(history)))
                return sequence[len(observed)-1]
            args=type('Args',(),{'run_dir':root,'model_dir':root,'ort_threads':1,'seed':31})()
            table=np.broadcast_to(np.zeros((1,768),dtype=np.float32),(6563,768))
            with patch.object(p,'_load_prepared',return_value=({},{})), patch.object(p,'_assemble_t3_embeddings',return_value=(np.zeros((1,4,768)),None)), patch.object(p.np,'load',return_value=table), patch.object(onnx_runtime,'_sample',side_effect=sample), patch.object(onnx_staged_runtime,'T3UnifiedOrtRuntime',Runtime), patch.object(p,'MAX_GENERATED_TOKENS',limit):
                result=p._run_t3_generation(args)
            return observed,result,np.load(root/'tokens/speech_tokens.npy')
    def test_raw_special_history_and_decode_progress(self):
        calls,result,tokens=self.invoke_tokens([p.SPEECH_BOS,42,p.SPEECH_EOS])
        self.assertEqual(calls,[(1.,[p.SPEECH_BOS]),(2.,[p.SPEECH_BOS]),(3.,[p.SPEECH_BOS,42])])
        self.assertEqual(tokens.tolist(),[42,p.S3GEN_SIL,p.S3GEN_SIL,p.S3GEN_SIL])
        self.assertFalse(result['length_limit_reached'])
    def test_no_eos_fails(self):
        with self.assertRaisesRegex(RuntimeError,'without EOS'):
            self.invoke_tokens([42,43],limit=2)
    def test_odd_reference_mel_length(self):
        # Harvey has 177 reference tokens but 355 reference mel frames.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'flow').mkdir()
            (root/'flow/config.json').write_text(json.dumps({'speech_token_count':135,'prompt_feat_length':355}))
            np.savez(root/'flow/flow_inputs.npz',mu=np.zeros((1,80,624),np.float32),mask=np.ones((1,1,624),np.float32),cond=np.zeros((1,80,624),np.float32),speaker_embedding=np.zeros((1,192),np.float32))
            class Runtime:
                def __init__(self,*a,**kw): pass
                def estimate(self,x,*args): return np.zeros_like(x)
            args=type('Args',(),{'run_dir':root,'model_dir':root,'ort_threads':1,'steps':2,'seed':31})()
            with patch.object(onnx_staged_runtime,'MeanflowEstimatorOrtRuntime',Runtime):
                p.stage_estimator(args)
            with np.load(root/'estimator/mel.npz') as values:
                self.assertEqual(values['mel'].shape,(1,80,269))
                np.testing.assert_array_equal(values['mel'],values['noise_speech'][:,:,1:])
    def test_master_matches_existing_policy(self):
        from scipy import signal
        import pyloudnorm as ln
        source=Path(__file__).with_name('quality_sweep.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='master')
        scope={'np':np,'signal':signal,'ln':ln}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<master>','exec'),scope)
        audio=np.random.default_rng(7).normal(0,.03,24000).astype(np.float32)
        expected,emeta=scope['master'](audio)
        actual,ameta=p._master_light(audio)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(ameta,emeta)

if __name__=='__main__': unittest.main()
