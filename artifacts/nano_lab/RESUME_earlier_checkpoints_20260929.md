# Current checkpoint: decoder fit and cadence work

Active model job: exec session88464, guarded `fit_decoder_embedding.py --max-steps40 --patience3`, log `artifacts/nano_lab/decoder_embedding_fit.log`, output `decoder_embedding_fit`. At step22 validationtotal .90273 vs baseline1.18127; cosine .98175. No finished fit claim yet. All agents interrupted. One job only.

Completed this pass:
- `decoder_embedding_comparison`:12 new-text native clips, independent expressive donor genembedding variants; all12SmallWER0; tokens exact pertext. MaxsampledprocessRSS2395.42MiB. Levelmatched-27LUFS, identity/DNS/prosody complete; review.json andindex.html linkedfromdelivery. Baselinecos.8686DNS3.1773; halfembedding.8807/3.1217; fullembedding.8828/3.0014; fullgen.8422/3.1785. Fullgen pitch201.2->220.2Hz butidentityregresses. No defaultchange.9helpertestsPASS.
- `acoustic_conditioning_probe` earlier source-token diagnostic: selfconditions deliberate targetleak, notzero-shot. Genembedding-only movespitch butnotworderrors;promptonlyfixesworderrors. Source ref itselfstyles lowerpitch.
- `cadence_penalty`:9nativeclips(heldout01seed31/question79/narrative97 xRP1.0/1.1/1.2), RSS2347.72MiB. Small8/9exact; RP1.1questionWER.1875 rejected; RP1.0all3exact. Durations1.0vs1.2:7.2vs6.6,6.32vs4.96,7.72vs6.88. listening_inputs.json andprosody_inputs.json prepared, butlevelmatch/DNS/prosody pending.
- New `scripts/nano_lab/fit_decoder_embedding.py` +8tests+doc. Flowonly~437MiBweights, frozen;192DembeddingMSE+envelope+cosreg, normpreserved,cossphericalcap>=.98;15train2valid. Strictparitybeforefit. Fixedmissing3SIL, nativeoptionalprompt_feat_lenNone, exactperrowseeds, epoch0checkpoint/progressreport/frozenweightgradchecks.
- Criticalnoisebug caughtbeforetraining: nativeS3Token2Wav.flow_inference draws TARGETnoisefirst, thenCFMfullmu noiseandsplicesfirsttargetsuffix. Helperoriginalsinglefulldraw failedparity huge. Corrected two-draworder. Differentialpre-fix artifact `decoder_flow_parity_probe.json` shows exactmanual-vsdirectCFM, butnotfullentrypoint.
- `decoder_embedding_fit_smoke`failedoptionalNone;v2failedwrongnoiseparity;v3PASSEDall17nativepreparedmelsexact(max0), onefitstepfinitegrad.2979/frozenparamsnograds, validation1.18127->1.16112,cossim.99995; peakRSS1707MiB. No tolerance relaxation. Failedlogs retained.
- `decoder_fitted_comparison_config.json` ready12cases=3texts xbase/fit embedding xmelcorrectionon/off. Usesfinalfitconditionals.pt onlyafterfitcomplete. Needgenerate, Small,levelmatch,evaluate,prosody; checktokenssame. Neitherthisfit norcadencepromoted.
- Runtimeagentverified actualstagedONNXpeakvocoder1004.27MiB vsprepare266.87MiB; no misleadingNumPyprepoptimization. RSS/QUALITYreports updatedforcompleteddecoderreference experiment.
- Hosthadunrelateddpkg-deb2.1GiB andavail4GiB; launchguardrefusedjobs, resumedafterheadroom8GiB. Neverkillunrelated/no cacheflush.

Next: pollactivefitthenfinishcandidateaudioauditsandcadencependingmetrics. Goalactive; user saidneithervoiceconvincing. No realismacceptance. PriorONNXdeliverynumbersremainvalidbelow.

# Latest checkpoint: fitted ONNX complete, realism remains unaccepted

No model job is running at this checkpoint. All agents are interrupted. Goal remains active. User's last listening feedback remains "Neither sounds convincing yet" for the original six samples. A later optional ASMR listening question has no reply. Do not mark the goal complete or claim professional realism.

## Delivered this pass

- Both fitted T3 ONNX roots exist: `onnx_staged_adapter_aligned_all_attn` and `onnx_staged_adapter_harvey_rank2`. Streaming patcher changed only 24 attention projection payloads. Peak cgroup memory 542/541 MiB. Baseline graph untouched.
- `verify_fitted_onnx.py` references: `fitted_t3_onnx_reference`, `harvey_fitted_t3_onnx_reference`. Both L32/L128 prefill + one-token saved-cache decode pass CPU and CUDA at preset 3e-4. CPU result copies are `verification_cpu.json`. CUDA reports are current `verification.json`; stage manifest `fitted_adapter_verification` contains graph/report hashes and scope. Explicit verifier-only loader solves bootstrap without fake verification.
- CUDA initially failed prefill with default TF32 (max .0558). Set `_cuda_provider_options.use_tf32=0`; CUDA maxabs becomes ASMR4.01e-5/Harvey4.20e-5. Failed ASMR report preserved `verification_cuda_tf32_default_failed.json`. The separate base context400 CPU cache mismatch remains unresolved. Do not claim all-context parity. `--experimental-t3` required.
- `_validate_adapter_provenance` checks graph+checkpoint adapter hash/scale. Base profiles reject fitted graphs. Fitted profiles reject base/mismatched graphs. Mel correction loaded and validated, applied only immediately before vocoder, recorded in metadata.
- `file_fingerprint.py` bounds process-local graph hash cache to64 entries with resolvedpath/dev/inode/size/mtime/ctime invalidation and stream-read race check. Caches no weights. Both runtime and pipeline use it only for graph hashes.
- Pre-cache full-FP32 three-request batches: `onnx_batch_asmr_fitted_fp32` and `onnx_batch_harvey_fitted_fp32`. All6 Small transcripts exact in `fitted_onnx_small_audit.json`.
- Post-cache final batches: `onnx_batch_asmr_fitted_cached`, `onnx_batch_harvey_fitted_cached`. ASMR treeRSS1936.96MiB, batch49.14s, warmfull3.2396/5.48audio and3.8669/6.52audio. Harvey treeRSS1939.57, batch46.84s, warmfull3.1574/6.54audio and2.7080/4.10audio. Full request includes watermark/master/WAV, excludes pre-stage preparation and process startup. Workers exit after batches.
- Before/after same manifests: ASMR warm5.58/6.21 ->3.24/3.87; Harvey4.55/4.27 ->3.16/2.71. Each `cache_parity.json` confirms tokens+mel EXACT; finalWAVdiff <=3.58e-6. Do not claim final WAV exact or direct post-cache ASR audit; delivered listening page uses the pre-cache audited WAVs.
- `onnx_asmr_fitted_staged_fp32/run.json`: explicit low-memory launcher succeeds, treeRSS1004.27MiB,42.419s cold for5.48audio. Tokens+mel exact corresponding batch; WAVmax3.46e-6. `fitted_onnx_staged_batch_parity.json`.
- `fitted_onnx_samples/index.html`: 4 unique audited generated clips +2sources, constantgain-27LUFS. Central delivery/index.html and QUALITY_REPORT/RSS_REPORT/voicesREADME updated. All links verified, no browser available for rendered visual inspection.
- 35 pure tests passed across fingerprint,verifier,patcher,profile,pipeline,batch,prosody. No model workloads in agents. Keep bounded_job guard serial.

## Quality findings and likely next investigation

- `prosody_audit.py` uses Praat pitch60-500Hz with explicit breathy/whisper reliability caveat; sentinel-200 HNR frames excluded, defined coverage recorded. No quality score.
- `asmr_same_text/prosody_audit.json`: source10s, median voicedF0258.9Hz (32%coverage), base5.96s/193.5Hz, combined6.8s/194.0Hz(41%coverage). Timing and pitch remain mismatched; no automatic pitchshift justified from oneclip.
- `codec_diagnostics/prosody_audit.json`: separate6s source253.1Hz, source-mel vocoder252.6Hz, native source-token reconstruction221.1Hz, olderORTtoken reconstruction222.9Hz. Source tokens/reference conditioning confounded; investigate token-to-mel/prosody. Do not blame all F0 issues on tokenizer or multiply all-unvoicedHiFTF0 output. Source-mel vocoder preserves pitch despite known F0branch behavior.
- ASMR data audit: dataset_repair_complete17rows uses fullsourceTiny proposals vsclipSmall exactagreement. Stricter clipTinyrebase dataset_repair_clip already exists and admits15, notmore. Bothcandidate-only34rowauditfiles exist; completeaudit36rows includes2calibration rows and cannot directlyfeedrebase. No readyverified expansion; protectedintervals/confidencegates muststay.
- Best native fitted samples still await user listening. New ONNX export does not prove new voice-quality gain; it preservesfitted weights with lowerRSS. Both speaker-specificfits are NOT generalzero-shot modelimprovement.

## Previous checkpoints

# Immediate live work

NO MODEL JOB ACTIVE after exec49987 Harvey matched-level DNS (finished). Three code-only workers: onnx_staged builds streaming patch_onnx_adapter.py; runtime_engine integrates model_dir-aware fitted-profile hash gates+mel calibration intoonnx_pipeline; adaptation builds verify_fitted_onnx.py Torch-reference and separatepureORTverify commands. Interruptfinishedchildren. Parentshouldreviewthenrunpatchserializedguard --small-job, reference/verifyseparatejobs, twofittedONNXvoicesaudio+RSSbenchmarks. Baselinegraphmuststaybyteunchanged.
NativebothfittedCLItested: ASMR2373.29MiB;Harvey2400.19MiB; BOTHbyteidenticalauditedsweepmasters. Newprofilesasmr_fitted_experimental andharvey_fitted_experimental stillawaitinglisten, notzero-shot. Nativeclone nowalsoincludesprofile+adapterSHAmetadatafuturejobs.
LatestintegratedONNX3outputs ALLSmallASRexact (current_final_small_audit.json), same-textASMR3ALLSmallASRexact. ASMRsame-text sourceheldout01verifiedSmallagreesoriginalTiny, excludedfit. Threegeneratedvariants base/adapter/combined nowasmr_same_text incl-27LUFScopies.
Updateddelivery/index.html shows6players ASMRsource/samewordsbase/combined,Harveyisolatedsource/businessbase/full; previouspagearchivedindex_initial_rejected.html.13linkedassetsexist. NoavailableCUAbrowser so renderingnotverified. Noauditoryacceptanceclaimed.
Harveyisolatedheldoutnewactualreferencepathsverified: meanbasecos.798861 speaker.808598 half.797678 full.818732. Atmatched-27LUFS DNScleanliness base3.3991 speaker3.4530 half3.4361 full3.3788 -> fullidentitygainhas SMALLnegativecleanlinessscoretradeoff, notuniversalimprovement. review+QUALITYupdated.
IntegratedbaseORTfullwarmrequest4.323s/7.36audio and3.844s/6.96audio, TREEpeak1942.91MiB;worker1811.75,cgroup1.4G. Coldfirst30.161s; whole3batch52.40s. RSSreportupdated. Tokens/melEXACTpriorORT-onlybatch; waveformmaxdiff3.58e-7; noaudioqualityclaim.
Patcherdesignconfirmed24inline FP32weights namedt3.tfmr.h.{0..11}.attn.c_attn.weight[768,2304], c_proj[768,768]. HFConv1DW[in,out];deltaA.T@B.T*alpha/rank*scale. Graph385854610bytesINLINE notexternal. Streamraw_data replacements preserveotherbytes. All-layerASMRrank4alpha8,Harveyrank2alpha4. Validatepatchhashprovenancebeforeallowingprofile. Runtimeworkercoordinatesmanifestadapter_patchschema.

# Current live checkpoint

Active goal remains unfulfilled: user rejected old6 ASMR samples, realism not established. New asyncquestion about all_attn_comparison/index.html pending. One modeljobexec6861 reference-only Harvey evaluator, correctedisolatedmanifestvariants path; --small-job --no-asr --no-dnsmos, logharvey_fitted_comparison/evaluation_isolated_reference_corrected.log.
Agent onnx_staged READONLY mapping fittedLoRAcheckpoint toONNXinitializers; onlyactivechild, interruptwhenfinished. Allothersinterrupted.
Pending queue: ASMR same-text sourceheldout01 generation configasmr_same_text_config.json; independentSmallauditof3generated + latestintegratedONNX3 (nativefittedCLIexactexistingauditedWAV, noextraASRneeded). EvaluateHarveyconstantlevelDNScopiesifneeded. Deliverupdatedpages/report. NeedONNXfittedadapterexportbeforefittedprofilelowRSSsupport; _load_profile nowrejects adapter/mel topreventsilentignores.
Native ASMR experimentalprofile added andSMOKEverified: native_asmr_fitted_question_79.wav BYTEIDENTICAL all_attn_robustness/question_79_combined.master.wav, RSS2373.29MiB,guardcgroup2.4G. Profileasmr_fitted_experimental statusawaitinglisten, modefittednotzero-shot. clone optionalmelhook+metadata.
ONNXpersistentbatch implemented onnx_batch.py factoryhook inonnx_pipeline.py. FirstORT-onlybatch CUDA3cases cold21.06s, warm4.21s/7.36audio and4.13s/6.96audio, treeRSS1504.79MiB,total64.96s (finishchildren24.12s). Tokens/mel exactstagedbaseline; floatvoc~8e-8diff. Latest integrated --finish-in-worker retainsCPUPerth SAME worker: onnx_batch_cuda_finish success52.40stotalbatch inclprepare10.86; cold FULLrequest30.161s; warmFULL4.323s/7.36audio and3.844s/6.96audio inclwatermark/master/WAV. PeakTREE1942.91MiB,workerpeak1811.75,cgroup1.4G,guard3GiB. Tokens/mel exactpriorbatch; WAVmaxdiff<=3.58e-7 afterinference_mode. T3strictlongcachegateunresolved, experimentalflagstillrequired. integratedSmallAUDITpending. Workerclosesafterbatch.
Harveystrictdatarepairdone: clipTiny+Small exactrows1,4,5 accepted,2train1valid. Cacheadaptation_harvey_aligned_cache EXACTsavedconditionals. rank2allattnKL2max24fit best4.5407epoch12; speakerfit best4.5059epoch3. 12caseharvey_fitted_comparison generated/evaluated allSmall12/12exact. Meanmixedheldoutcos base.722054 speaker.728552 half.709727 full.734738; DNSbase3.3799 full3.4230. Notpromotedtinydata. Listeningpageindex.html usesconstantgain-27LUFScopiesharvey_level_matched (14filesincl2sources), nofilters. Alternativeisolatedheldout scoringinitialtry changedfilesbut_evaluatorusesvariants -> preservedinvalidalternative as evaluation_files_mapping_only.json, fixedmanifestvariants. Currentexec6861reruns.
ASMRallattnrobustness12cases (2texts*2seeds*base/adapter/combined) complete. Combinedcos.874520 vsbase.866843, DNS3.2393vs2.979. SmallALL12exact; Tinyquestion79as-auncertainty butSmallcorrect. Allattn3textscos.889739 vsbase.881681, DNS3.0437vs2.6154, SmallALL6exact.
ImportantLOUDNESScontrolimplemented level_match.py: constantgainonlyshared-27LUFS,4xpeakceiling-1dBTP no limiter/filter, remeasureerror<1e-6LU. 36ASMRfiles includingreference inlevel_matched. DNSatmatchedlevel 3textbase2.4818 allattn3.0669; robustnessbase2.8768 adapter3.1866 combined3.2347. Thusgainpersistslevelcontrol. ASMRall_attn_comparison/mel_comparison pages usematchedcopies. Allpageaudio/linksverifiedexists; CUAfoundnoavailablebrowser,sovisualrendernotverified.
Rebase supportrepair_dataset.py and audit_asr_ct2.py --word-timestamps implemented/tested12tests. Parentfixedworkeraccidentalperwordconfidencegate toexistingMEANwordconfidence>=.35 semantics; exactTiny+Smallstillmandatory. Originalreport/labels/confidences/hashes retained. ASMRclipTinyrebaseadmits15 (13train2valid), lowerthanoriginal17; notfitted/adopted. twoASMRcalibrationrows filteredfromaudits viahashedsubsetprovenance. Mainretains17aligneddataset.
HeldoutsourceSmallchecks: ASMR01 exact originaltext,04/05 originaltextincomplete soonly01 usednewsame-textconfig. Noheldoutsourceaudio entersfit. All3firstONNXbatchSmalltranscriptsexact.
QUALITY_REPORT.md rewritten currentaccurate, oldretainedQUALITY_REPORT_history.md. RSS_REPORTappendlatestORT-onlynumbers butintegratednumbersnotyetadded. Needcurrentdeliverypageindex linknewpages, docnewenginebenchmark/smoketest.

# Latest active work

No active main model job after tokenizer_gain_probe Small audit (exec95265 finished).
Threecode-onlychildren: adaptation writesnewmel_calibration.py; reference_quality generalizes repair_dataset.py family+referenceprovenance; runtime_engine addsoptionalactualORTkernelprofiling afterCUDAproviderplumbing. Interruptchildrenwhenfinished.
ALIGNEDASMR comparison9clipscompleted: basecos.88168DNS2.615; speaker.86437DNS3.104Small2/3exact; rank4.86603DNS3.292Small3/3exact. Neitherpromoted. aligned_comparison/index.html. Trainingreference nowexactinference tensors (aligned_conditioning_parity.json). load_cache rejectsoldconditioningprovenance.3datasetcontracttestsPASS.
Source1diagnosticCPUORT reconstruction SmallWER.3333,Tiny.4167 BUT nativeGPU rawtokens(same150IDs) atnoiseSeed10031 SmallWER0. So noirreversibletokenizerlossclaim; backend/noiserealizationconfounded. Normalizedtargetaudiochanges9tokens andSmallWER.0833, so targetnormalization NOTadopted. tokenizergainprobeRSS2334.9MiB cgroup1.7G. Rawtargets unchanged.
Current next queue: reviewHarveyfamilycode thenboundedTinyproposal(raw40.8secsource, protectedmanifest harvey_repair/protected_manifest.json);Smallauditstrictconsensus smalltrainingdiagnosticmayuseexplicitmintrain2+minvalid1 withshortsourcelimitation. Selectedref9-16.1, heldout34-40.8protected. ORTGPUfirstsmokeafterproviderprofilingdone, `.venv-nano-ort` mustuseguard3GiB andactualCUDAkernelprofilingevidence. Melcalibrationprepare17cleanalignedsourceclips withfixeddecoder cachedconditions GPUguard3GiB, thenpurefitonlytrain, heldoutgate, neverpromoteonlossalone.

# Latest live state

Prior comparison completed15files, matched3texts x5variants. review.json shows no promotion. Small audit14/15exact, halfadapter says blend instead of blue. Speakeradjustment improvedDNSMOSmean2.615->3.050 withcos.8817->.8760. Fulladaptercos.8907DNS3.011Small3/3exact butTiny2uncertain.
IMPORTANT second trainingbug confirmed: raw referencepreprocessing vs inference norm/resampling causes37/333differentprompttokens andspeaker cosine.9653. adaptation.py nowcalls exactmodel.prepare_conditionals onceperreference. Parentfixedprovenanceexaggeration.5 (Nano ignoresit). Newalignedprepare running exec73950,logaligned_prepare.log. Outputadaptation_asmr_aligned_cache.json. Afterdone runcheck_conditioning_cache.py CPUcap640 to compare actualsavedcond. Requireexact; thenfreshspeaker andrank4KLfits, newalignednames.
Childadaptation interruptedaftercode. Childonnx_stagedinterruptedafterread-onlyF0investigation. Noothersactive.
Harvey vocoderdiagnostic done,cgroup518MiB: F0max.7627,allUV justlikeASMR. Noverifiedloaderbug; don'tmultiplyF0. Finalmelconvcanproducepitchindependently. Reportcodec_diagnostics/f0_investigation.json.

# Current update

Small CT2 audit completed under 1280 MiB cap; cgroup peak946.9MiB.
Strict dataset admission accepted17 clips:15train+2valid. New manifest
artifacts/nano_lab/dataset_repair_complete/adaptation_repaired.json.
No human verification, no production quality claim. Old fitted variants rejected.
Preparation running guarded3GiB, exec2548, log repaired_prepare.log.
Next fresh constrained residual + rank4KL adapter, then actual audio validation.

# Immediate live state

Tiny proposal watcher exec70413 is active for up to120s, then mayrun guarded
repair_dataset.py propose under1024MiB. Poll before anyothermodeljob. It needs
5120MiB available; firstprobe5081. Log repair_dataset_propose.log.
Child onnx_staged interrupted. repair_dataset.py reviewed andcompilepassed.
Parent fixed worker's non-strict0.20consensus default to0.0; nonzero threshold
is now rejected. Removed merging across omitted shortspeech. Sourcehash,
candidateaudiohash, protectedinterval gates added. Threepuretests pass.
Worker audit command is `repair_dataset.py stage --proposal ... --audit ...`.
Proposal defaults artifacts/nano_lab/repaired_dataset (checkconstant). After
propose, run audit_asr.py underdefaultguard, thenstagepureadmission. Need8train
plus2valid. Cache preparation/training nowenforces transcript_audit+hashes.

OldTinytrainingauditfinished. training_label_review.json joinsTiny+Small.
Bothrecognizers confirm serious cachedtext mismatch. Corrected source01TTS
nowgenerated; codec_diagnostics/index.html updated with agreedsourcewords.
New evaluation manifest: codec_diagnostics/evaluation_manifest_corrected.json.
Its corrected TTS audio has notbeen ASR-checked yet. Original faulty evaluation
retained andmarked invalid expectedlabels.

# QUALITY PRIORITY UPDATE

User rejected ASMR samples as unconvincing. Source reconstruction revealed bad
training labels: source01 says different words than cache. Small ASR matches
only3/10 cached labels exactly. Rows1,6,8,9,valid04 materially mismatched.
Old adapters and speaker residual results cannot justify adoption. Original
cache/checkpoints preserved, but dataset_contract.py now rejects them for
training without accepted text/audio-hash audits. Two regression tests pass.

CURRENT jobs: Tiny audit old training audio session50152 (poll), log
training_audio_tiny_audit.log; no other models. Child /root/onnx_staged CODE ONLY
writing repair_dataset.py: full source Tiny word timestamps, protected intervals
excluded, candidates3-12sec, Small audit strictconsensus admission >=8train+2val.
Interrupt child immediately whendone, review, then bounded propose/audit/admit.

Isolated .venv-nano-ort installed onnxruntime-gpu1.26.0 (CUDA12.8 per official
docs). .pth sharesCPU Torch and existingbackendNVIDIAdeps. No GPU ORT model
executed yet; existingenvs unchanged. Prioritize dataset repair now.

codec_diagnostics/index.html has source/vocoder/tokens clips fromsource01;
its TTS sample useswrongcachedtranscript. Page explicitly warns notsame-sentence.
Tiny source andvocoder agreeexact words, tokenreconstructionhasminorASRerror.
Do NOT call0.89WERtrue reconstructionfailure: expectedlabels wrong!

# Current state

Goal active. User priority: avoid desktop crashes. Use bounded_job.py for every
model, synthesis, training, export, and evaluation job. One job at a time.
Read scripts/nano_lab/AGENTS.md. Never raise limits or stop unrelated processes.
Check the systemd unit and live RAM before launching work.

## Completed

- Six candidates and source comparisons: delivery/index.html. New experimental
  conversational speaker residual at 75 percent. Zero-shot defaults unchanged.
- Actual soft and Harvey CLI outputs exactly match delivery hashes. Observed
  peak host RSS 2126.4 and 2136.2 MiB; peak GPU allocation reduced by 712 MiB.
- Round4 and robustness sweeps, independent Whisper Small audits completed.
  See round4_review.json and robustness_review.json. Full-strength residual
  had a content error. 75 percent is an experimental compromise, not a proven
  realism winner. Human listening feedback remains pending.
- Actual 85-word CLI passed with 32-word maximum chunks. 29.14 seconds audio,
  13.60 seconds generation/mastering, peak RSS 2473.3 MiB. Small ASR confirms
  content apart from ten versus 10 formatting. Forced length-limit retry also
  passed independent Small ASR with no errors. Atomic WAV publication retained.
- CPU-only Torch environment .venv-nano-cpu enables smaller evaluation jobs.
  It leaves .venv-nano and its GPU dependencies unchanged.
- Streamed external-parameter ONNX export works. Flow and meanflow estimator
  pass separate ORT checks across saved lengths. Short vocoder graph passes.
- Extended T3 reference generated successfully with CPU Torch, peak RSS
  1081.1 MiB. ORT strict 400-token prefill cache check FAILS: 176 elements of
  7,379,363 exceed combined 3e-4 tolerance, max abs 0.000989. Logits and decode
  cases pass. Disabling graph optimizations does not fix it. Keep experimental.
- 18-case vocoder noise sweep completed. Six groups have identical speech-token
  payloads across noise scales 0, 0.5, 1.0. Evaluation currently running.

## Latest execution

- Complete staged ONNX soft and Harvey runs now succeed under 1280 MiB cap.
  Soft complete: 7.36s audio /81.27s total, largest child833.2MiB RSS.
  Harvey fixed: 5.38s /67.71s, sampled process-tree peak807.9MiB.
  CPU latency is far from real-time; no default replacement.
- Parent fixed odd reference mel length: noise-tail insertion offset differs
  from output crop offset. Four pure pipeline tests pass.
- Soft acoustic and complete outputs pass Tiny and Small ASR, zero errors.
  Harvey evaluation pending.
- All 18 noise-sweep WAVs match within scale groups. Full soft native vocoder
  diagnostic has zero voiced frames (pitch predictor output max0.954, below
  model threshold10). This is not measured physical pitch of the speech.
- Full 7.36s vocoder native/ORT parity passes: maxabs1.049e-5, relL2 7.723e-6.
- User answered listening question: 'Neither sounds convincing yet'. Saved in
  delivery/listening_feedback.json. ASMR quality remains rejected. A follow-up
  asks whether identity, robotic prosody, or noise dominates; no answer yet.
- All child agents interrupted. New codec_diagnostics.py reviewed, compiled.
  Genuine tokens exported for asmr7_train_01/02. Source01 mel extracted (6s).
- CURRENT native source01 vocoder job: exec session pending in active thread,
  log codec_source01_native.log. Poll before another model job.

## Next

1. Complete source01 native vocoder, render it through pipeline finish to keep
   Perth/master. Generate source-token acoustic reconstruction using explicit
   diagnostic labels. Compare source, vocoder-only, token reconstruction, TTS.
2. Repeat second source if needed and evaluate actual delivered diagnostics.
3. Test all-source excitation noise only if the diagnostics support it. Current
   voiced-only control is ineffective; no default change.
4. Evaluate Harvey ONNX content, update runtime reports/commands.
5. Address realism based on source-stage loss and user's listening feedback.

Main reports: QUALITY_REPORT.md, RSS_REPORT.md, EXPERIMENT_DECISIONS.md.
No claim of ElevenLabs equivalence or perceptual realism is supported.
