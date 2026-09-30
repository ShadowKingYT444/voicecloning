"""Sketch of a microphone adapter: yield 16 kHz float32 chunks from your device."""

from whisper_fast_vad import WhisperFastVAD


def consume(microphone_chunks):
    stt = WhisperFastVAD(compute_type="int8", cpu_threads=1)
    for event in stt.stream_transcribe(microphone_chunks, sample_rate=16_000):
        if event.text:
            print(event.text)
