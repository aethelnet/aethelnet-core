# Security Policy

Please **do not open a public issue** for security problems.
Use **Security → Report a vulnerability** on this repository (GitHub private reporting).

Notes for users of this package:

- `LiquidGraph.load_state()` uses `torch.load(..., weights_only=False)`, which can execute code embedded
  in a checkpoint. Only load checkpoints you created yourself.
- The demo server in `aethelnet/engine.py` only checks `X-Auth-Token` when `AETHELNET_AUTH_TOKEN`
  is set, and its save/load endpoints accept file paths. Don't expose it to the internet without a token.
