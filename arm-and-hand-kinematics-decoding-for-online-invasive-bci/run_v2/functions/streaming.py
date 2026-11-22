"""A single-bin API for saved Ridge, Kalman, GRU, LFADS-lite, and Transformer models.

The decode flag gates output and resets per-attempt state. Pickles/PT files must
come from this trusted training package. This measures computation, not Redis I/O.
"""
from pathlib import Path
import pickle
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from brand_streaming_decoder_node import StreamingQposDecoder, StreamStep, VM_BODY_ID_TO_COMPACT_OBJECT_ID
from run_split_pca_model_suite import kalman_gains


class Decoder:
    def __init__(self, artifact, initial_qpos=None, device='cpu'):
        self.initial = np.zeros(32, dtype=np.float32) if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float32)
        self.deep = None
        artifact = Path(artifact)
        if artifact.suffix == '.pt':
            self.deep = StreamingQposDecoder(artifact, device=device, max_context_bins=4096,
                                             initial_qpos=self.initial)
            self.neural_raw_dim = self.deep.neural_raw_dim
            return
        with artifact.open('rb') as f:
            self.payload = pickle.load(f)
        p = self.payload
        components = p['projector']['components']
        if components is not None:
            self.neural_raw_dim = int(np.asarray(components).shape[1])
        elif p['model_family'] == 'ridge':
            self.neural_raw_dim = len(p['x_mean']) - 1 - (p['n_objects'] if p['condition_object'] else 0)
        else:
            self.neural_raw_dim = int(p['params']['H'].shape[0])
        self.reset()

    def reset(self):
        self.active = False
        self.index = 0
        self.ema = None
        self.output = None
        self.previous_velocity = None
        self.qpos = self.initial.copy()
        if hasattr(self, 'payload') and self.payload['model_family'] in {'kalman', 'refit_kalman'}:
            self.z = self.payload['params']['z_mean'].copy()

    def step(self, neural, *, decode_enabled, timestamp_s, object_id=None):
        if self.deep is not None:
            return self.deep.step(neural, decode_enabled=decode_enabled,
                                  timestamp_s=timestamp_s, object_id=object_id)
        if not decode_enabled:
            self.reset()
            return None
        if not self.active:
            self.reset()
            self.active = True
        object_id = VM_BODY_ID_TO_COMPACT_OBJECT_ID.get(object_id,object_id)
        start = time.perf_counter()
        x = np.asarray(neural, dtype=np.float32).reshape(-1)
        if x.shape != (self.neural_raw_dim,) or not np.isfinite(x).all():
            raise ValueError(f'Expected {self.neural_raw_dim} finite input features')
        self.ema = x.copy() if self.ema is None else .15*x + .85*self.ema
        x = self.ema
        p = self.payload
        projector = p['projector']
        if projector['components'] is not None:
            x = (x-projector['mean']) @ projector['components'].T
        phase = self.index / 600.
        if p['model_family'] == 'ridge':
            parts = [x, np.array([phase], dtype=np.float32)]
            if p['condition_object']:
                if object_id not in range(6):
                    raise ValueError('Object-conditioned model needs compact object ID 0..5')
                cue = np.zeros(p['n_objects'], dtype=np.float32)
                cue[object_id] = 1
                parts.append(cue)
            feature = np.concatenate(parts)
            y = (((feature-p['x_mean'])/p['x_scale']) @ p['weights'] + p['bias']) * p['y_scale'] + p['y_mean']
            if p['target_components'] is not None:
                y = y @ p['target_components'] + p['target_mean']
            alpha = p['output_alpha']
            self.output = y.copy() if self.output is None else alpha*y + (1-alpha)*self.output
            y = self.output
        else:
            params = p['params']
            prior = params['A'] @ self.z + params['A_b']
            gain = kalman_gains(params, self.index+1)[self.index]
            self.z = prior + gain @ (x - (params['H'] @ prior + params['H_b']))
            state = self.z
            if p['state_components'] is not None:
                state = state @ p['state_components'] + p['state_mean']
            y = state[32:64] if p['target_base_mode'] == 'qvel' else state[:32]
        if p['target_base_mode'] == 'qvel':
            if self.index > 0:
                self.qpos = self.qpos + self.previous_velocity / 100.
            self.previous_velocity = y[:32].copy()
        else:
            self.qpos = y[:32].copy()
        self.index += 1
        return StreamStep(timestamp_s, self.qpos.copy(), phase,
                          (time.perf_counter()-start)*1000, object_id, object_id)
