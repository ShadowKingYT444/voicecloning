"""Evaluate a generated sweep without importing its synthesis runtime."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
import argparse,json
from pathlib import Path
from reference_evaluator import evaluate, DEFAULT_WHISPER_MODEL
from dnsmos import DNSMOS

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--no-asr",action="store_true");p.add_argument("--final-audit",action="store_true")
    p.add_argument("--no-dnsmos",action="store_true",help="skip unchanged audio-quality scoring during reference-only checks")
    p.add_argument("--references",type=Path,default=ROOT/"artifacts/nano_lab/references/manifest.json")
    a=p.parse_args();source=json.loads(a.manifest.read_text())
    rows=source if isinstance(source,list) else source["runs"]
    candidates=[{"label":r.get("id",Path(r["path"]).stem),"audio_path":r["path"],
                 "family":"asmr7" if r.get("voice","asmr7") in ["asmr","asmr7"] else "harvey",
                 "expected_text":r["text"]} for r in rows if "path" in r and "text" in r and not r.get("error")]
    refs=json.loads(a.references.read_text())
    excluded={"asmr7_heldout_01","asmr7_heldout_04"}
    if a.final_audit: excluded|={"asmr7_heldout_02","asmr7_heldout_03"}
    else: excluded.add("asmr7_heldout_05")
    report=evaluate(refs,candidates,whisper_model=DEFAULT_WHISPER_MODEL,transcribe=not a.no_asr,
                    speaker=True,heldout_variant="natural",excluded_heldout_ids=excluded)
    report["synthesis_manifest"]=str(a.manifest.resolve())
    report["reference_manifest"]=str(a.references.resolve())
    report["manifest"]=str(a.references.resolve())
    report["dnsmos_enabled"]=not a.no_dnsmos
    report["evaluation_role"]="final_source_audit" if a.final_audit else "development_selection"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    if a.no_dnsmos:
        return
    dnsmos=DNSMOS()
    for r in report["inputs"]:
        r.update(dnsmos(r["audio_path"]))
        print(json.dumps({"id":r["label"],"cosine":r.get("speaker_similarity",{}).get("mean_cosine"),
                         "wer":r.get("wer_contraction_normalized",r.get("wer",{})).get("wer"),
                         "dnsmos_overall":r["dnsmos_overall"]}),flush=True)
        a.out.write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
