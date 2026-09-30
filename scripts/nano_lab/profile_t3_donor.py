"""Pinned T3-only donor caches for native and staged voice profiles."""
from pathlib import Path
from types import SimpleNamespace

from file_fingerprint import fingerprint_file
from t3_conditioning_donor import VALID_MODES, copy_t3_conditioning


def resolve_t3_donor(profile, root):
    value = profile.get('t3_donor_cache')
    if not value:
        if profile.get('t3_donor_mode') or profile.get('t3_donor_cache_sha256'):
            raise ValueError('T3 donor metadata requires t3_donor_cache')
        return None
    mode = profile.get('t3_donor_mode', 'prompt')
    if mode not in VALID_MODES:
        raise ValueError('Invalid T3 donor mode')
    path = (Path(root) / value).resolve()
    expected = profile.get('t3_donor_cache_sha256')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('T3 donor profile must pin its cache SHA-256')
    if not path.is_file():
        raise FileNotFoundError(path)
    if fingerprint_file(path) != expected:
        raise ValueError('T3 donor cache SHA-256 mismatch')
    return dict(path=str(path), sha256=expected, mode=mode)


def copy_t3_payload(donor, destination, mode):
    """Mutate only a weights-only destination's T3 dictionary."""
    if not isinstance(donor, dict) or not isinstance(donor.get('t3'), dict):
        raise ValueError('Donor has no T3 dictionary')
    if not isinstance(destination, dict) or not isinstance(destination.get('t3'), dict):
        raise ValueError('Destination has no T3 dictionary')
    donor_view = SimpleNamespace(t3=SimpleNamespace(**donor['t3']))
    destination_view = SimpleNamespace(t3=SimpleNamespace(**destination['t3']))
    metadata = copy_t3_conditioning(donor_view, destination_view, mode)
    destination['t3'] = vars(destination_view.t3)
    return metadata
