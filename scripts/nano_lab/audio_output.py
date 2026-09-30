"""Small output helpers that do not import the speech model."""
import os, re, tempfile
from pathlib import Path
from contextlib import contextmanager
import soundfile as sf

@contextmanager
def completed_audio(path, sample_rate):
    """Publish the WAV only after every segment succeeds."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".wav", dir=path.parent)
    os.close(fd)
    try:
        with sf.SoundFile(temporary, "w", samplerate=sample_rate, channels=1, subtype="PCM_24") as writer:
            yield writer
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def chunks(text, max_words=55):
    sentences=re.split(r'(?<=[.!?])\s+',text.strip())
    current=[]
    for sentence in sentences:
        words=sentence.split()
        if len(words)>max_words:
            if current: yield " ".join(current); current=[]
            for i in range(0,len(words),max_words): yield " ".join(words[i:i+max_words])
        elif sum(len(s.split()) for s in current)+len(words)>max_words:
            yield " ".join(current); current=[sentence]
        else: current.append(sentence)
    if current: yield " ".join(current)

def split_failed_segment(text):
    """Bound a length-limit retry without dropping or changing input words."""
    words = text.split()
    if len(words) <= 12:
        return None
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) > 1:
        # Keep whole sentences when possible. A hard half-word budget can
        # otherwise turn a thirteen-word sentence into twelve words plus one.
        cut = min(range(1,len(sentences)), key=lambda i:abs(sum(len(s.split()) for s in sentences[:i])-len(words)/2))
        return [" ".join(sentences[:cut]), " ".join(sentences[cut:])]
    cut = len(words)//2
    return [" ".join(words[:cut]), " ".join(words[cut:])]
