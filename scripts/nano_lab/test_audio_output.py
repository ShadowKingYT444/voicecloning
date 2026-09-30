"""Verify publication and chunk boundaries without importing a speech model."""
import tempfile
import unittest
from pathlib import Path
import soundfile as sf
from audio_output import chunks, completed_audio, split_failed_segment


class AudioOutputTests(unittest.TestCase):
    def test_retry_keeps_complete_sentences(self):
        first="The rain has finally stopped."
        second="I opened the curtains and made a fresh cup of coffee."
        self.assertEqual(split_failed_segment(first+" "+second),[first,second])

    def test_length_retry_terminates_and_preserves_every_word(self):
        original="First sentence. "+" ".join(f"word{i}" for i in range(80))+". Last sentence."
        pending=[original]
        terminal=[]
        attempts=0
        while pending:
            part=pending.pop(0)
            smaller=split_failed_segment(part)
            if smaller is None:terminal.append(part)
            else:pending=smaller+pending
            attempts+=1
            self.assertLess(attempts,100)
        self.assertEqual(" ".join(terminal).split(),original.split())
        self.assertTrue(all(len(x.split())<=12 for x in terminal))

    def test_failed_segment_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"sample.wav"
            output.write_bytes(b"previous complete output")
            with self.assertRaisesRegex(RuntimeError,"failed"):
                with completed_audio(output,24000) as writer:
                    writer.write([0.,.1,-.1])
                    raise RuntimeError("segment failed")
            self.assertEqual(output.read_bytes(),b"previous complete output")
            self.assertEqual(list(Path(folder).iterdir()),[output])

    def test_success_publishes_complete_pcm24(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"sample.wav"
            with completed_audio(output,24000) as writer:
                writer.write([0.,.1,-.1])
                self.assertFalse(output.exists())
            info=sf.info(output)
            self.assertEqual((info.frames,info.samplerate,info.subtype),(3,24000,"PCM_24"))

    def test_long_input_preserves_order_and_bounds(self):
        text="First short sentence. "+" ".join(f"word{i}" for i in range(130))+". Last sentence."
        parts=list(chunks(text))
        self.assertEqual(" ".join(parts),text)
        self.assertTrue(all(len(part.split())<=55 for part in parts))


if __name__=="__main__":
    unittest.main()
