# Running the scripts

The benchmark and DSpark scripts import `mlx_lm` **as oMLX patches it** — stock
`mlx-lm` has no `deepseek_v4` architecture
([#1233](https://github.com/ml-explore/mlx-lm/issues/1233),
[#1281](https://github.com/ml-explore/mlx-lm/issues/1281)), so they must run
under oMLX's bundled interpreter with the `omlx` package importable.

```sh
export OMLX=/Applications/oMLX.app/Contents/Resources
export PY=$OMLX/Python/cpython-3.11/bin/python3.11
export PYTHONPATH=$OMLX:$OMLX/Python/framework-mlx-base/lib/python3.11/site-packages:$PWD

MODEL=~/.omlx/models/Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX
```

Stop the oMLX server first for anything that loads the model — the weights are
~146 GiB resident and two copies will not fit:

```sh
pkill -f 'oMLX|omlx-server'
```

## Examples

```sh
# baseline decode
$PY tools/bench_ds4.py $MODEL --max-tokens 256

# dense vs windowed prefill
$PY tools/bench_windowed.py $MODEL --lengths 4096,8192,16384

# DSpark acceptance on your own content
$PY ds4/acceptance.py $MODEL --prompt "$(cat some_file.py)"

# speculative decode vs greedy
$PY ds4/spec_decode.py $MODEL --max-tokens 192 --baseline

# is a k-token verify cheaper than k single steps?
$PY tools/multitoken_cost.py $MODEL
```

`tools/server_prefill.py` is the exception — it talks to the running server over
HTTP, so use any Python 3 and leave oMLX up:

```sh
python3 tools/server_prefill.py
```

It reads the API key from `OMLX_API_KEY`, or from `~/.omlx/settings.json`.

## Notes

- `tools/audit_quant.py` and `tools/fix_quant_keys.py` are pure stdlib and read
  only safetensors headers — any Python 3.7+ works, no oMLX needed, and neither
  loads a tensor.
- Scripts that load the model take ~20 s and ~146 GiB. A 256 GB machine is
  comfortable; 192 GB is not.
