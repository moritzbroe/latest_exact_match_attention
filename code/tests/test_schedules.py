"""CPU checks of the lr/c/alpha schedules in train.py. Run: python tests/test_schedules.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lema.train import TrainConfig, alpha_at, c_at, lr_at

cfg = TrainConfig(steps=22000, out="/tmp/x", lr=1e-4, lr_final=None, warmup=1000,
                  c_start=1000, c_steps=10000, alpha_speed=1000, alpha_target=10.0)
assert lr_at(cfg, 0) < lr_at(cfg, 500) < lr_at(cfg, 999)
assert lr_at(cfg, 1000) == lr_at(cfg, 21999) == 1e-4          # constant after warmup
assert c_at(cfg, 0, 63.0) == 0.0 and c_at(cfg, 6000, 63.0) == 31.5
assert c_at(cfg, 11000, 63.0) == c_at(cfg, 21999, 63.0) == 63.0
assert alpha_at(cfg, 0, 0.125) == alpha_at(cfg, 11000, 0.125) == 0.125   # flat until c-end
assert abs(alpha_at(cfg, 16000, 0.125) - 5.125) < 1e-9        # +1 per 1000 steps
assert alpha_at(cfg, 21999, 0.125) == 10.0

# the target only sets where the ramp STOPS: it never changes the trajectory before that
t5 = TrainConfig(steps=22000, out="/tmp/x", c_start=1000, c_steps=10000,
                 alpha_speed=1000, alpha_target=5.0)
for s in range(0, 22000, 250):
    assert alpha_at(t5, s, 0.125) == min(5.0, alpha_at(cfg, s, 0.125)), s

cos = TrainConfig(steps=20000, out="/tmp/x", lr=1e-4, lr_final=1e-5, warmup=1000)
assert 0.99e-4 < lr_at(cos, 1000) < 1e-4                      # whole-run cosine
assert abs(lr_at(cos, 19999) - 1e-5) < 1e-8
assert lr_at(cos, 10000) < lr_at(cos, 5000) < lr_at(cos, 1000)

# the recall recipe's lr switch once hardening starts
sw = TrainConfig(steps=22000, out="/tmp/x", lr=3e-4, warmup=1000, lr_alpha_phase=5e-5,
                 c_start=1000, c_steps=10000)
assert lr_at(sw, 10999) == 3e-4 and lr_at(sw, 11000) == 5e-5
print("test_schedules: OK")
