"""Independent Whisper-small content audit. This is not a listening test."""
import argparse,json,math,os,time,hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from reference_evaluator import word_error_rate,_expand_contractions

MODEL=Path('/home/terryd/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d')

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=parser.parse_args();torch.set_num_threads(2)
    rows=json.loads(args.manifest.read_text())
    if isinstance(rows,dict):rows=rows.get('runs',rows.get('inputs',[]))
    processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,low_cpu_mem_usage=True).to(args.device).eval()
    report={'model':str(MODEL),'role':'independent_content_audit','human_listening':False,'inputs':[]}
    for row in rows:
        if row.get('error'):continue
        path=row.get('path',row.get('audio_path'));expected=row.get('text',row.get('expected_text'))
        if not path or not expected:continue
        audio,sr=sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
        divisor=math.gcd(sr,16000);audio=resample_poly(audio,16000//divisor,sr//divisor)
        transcript=[];start=time.perf_counter()
        for offset in range(0,len(audio),25*16000):
            features=processor(audio[offset:offset+25*16000],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            tokens=model.generate(input_features=features.input_features.to(args.device),attention_mask=features.attention_mask.to(args.device),language='en',task='transcribe',do_sample=False,max_new_tokens=256)
            transcript.append(processor.batch_decode(tokens,skip_special_tokens=True)[0].strip())
        text=' '.join(transcript)
        with open(path,'rb') as handle: audio_hash=hashlib.file_digest(handle,'sha256').hexdigest()
        result={'id':row.get('id',row.get('label',Path(path).stem)),'path':path,'audio_sha256':audio_hash,'expected_text':expected,'transcript':text,
                'wer':word_error_rate(expected,text),'wer_contraction_normalized':word_error_rate(_expand_contractions(expected),_expand_contractions(text)),
                'seconds':time.perf_counter()-start}
        report['inputs'].append(result);args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'id':result['id'],'wer':result['wer_contraction_normalized']['wer'],'transcript':text}),flush=True)

if __name__=='__main__':main()
