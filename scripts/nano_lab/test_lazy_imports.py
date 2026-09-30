"""Verify lazy package imports without instantiating a voice model."""
import sys
import chatterbox

assert "chatterbox.tts" not in sys.modules
assert "chatterbox.mtl_tts" not in sys.modules
assert "torch" not in sys.modules

from chatterbox.models.s3gen.hifigan import HiFTGenerator
assert "chatterbox.models.s3gen.s3gen" not in sys.modules
assert "chatterbox.models.t3.t3" not in sys.modules

from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3gen.s3gen import S3Token2Wav
assert S3Gen is S3Token2Wav and S3GEN_SR == 24000

from chatterbox import ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.tts import ChatterboxTTS as DirectTTS
from chatterbox.vc import ChatterboxVC as DirectVC
from chatterbox.mtl_tts import ChatterboxMultilingualTTS as DirectMTL
assert (ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS) == (DirectTTS, DirectVC, DirectMTL)
assert SUPPORTED_LANGUAGES
print("PASS: component imports stay lazy and public API aliases retain identity")
