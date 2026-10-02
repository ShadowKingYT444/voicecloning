"""Shared deferred masks must preserve full and cached GPT-2 outputs exactly."""
import unittest
import torch
from transformers import GPT2Config, GPT2Model
from runtime import _defer_t3_causal_masks, _repair_t3_runtime_buffers


class MaskTests(unittest.TestCase):
    def test_meta_allocation_and_numeric_parity(self):
        torch.manual_seed(31)
        config = GPT2Config(vocab_size=32, n_positions=64, n_embd=24,
                            n_layer=3, n_head=3, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        config._attn_implementation = 'eager'
        stock = GPT2Model(config).eval()
        with torch.device('meta'):
            candidate = GPT2Model(config)
        _defer_t3_causal_masks(candidate)
        self.assertTrue(all(layer.attn.bias is None for layer in candidate.h))
        candidate.to_empty(device='cpu')
        candidate.load_state_dict(stock.state_dict(), strict=True)
        _repair_t3_runtime_buffers(candidate, torch.device('cpu'))
        candidate.eval()
        pointer = candidate.h[0].attn.bias.data_ptr()
        self.assertTrue(all(layer.attn.bias.data_ptr() == pointer for layer in candidate.h))
        tokens = torch.tensor([[1, 3, 5, 8, 4, 2]])
        with torch.inference_mode():
            left = stock(tokens, use_cache=True)
            right = candidate(tokens, use_cache=True)
            self.assertTrue(torch.equal(left.last_hidden_state, right.last_hidden_state))
            left_next = stock(torch.tensor([[7]]), past_key_values=left.past_key_values)
            right_next = candidate(torch.tensor([[7]]), past_key_values=right.past_key_values)
            self.assertTrue(torch.equal(left_next.last_hidden_state, right_next.last_hidden_state))


if __name__ == '__main__':
    unittest.main()
