#!/usr/bin/env python
import sys
if len(sys.argv) < 2: raise ValueError(f"Usage: {sys.argv[0]} path_to_ckpt")

from pathlib import Path
import torch as th
import numpy as np

ckpt = th.load(sys.argv[1], map_location="cpu")
outpath = Path("maple-diffusion/bins")
outpath.mkdir(exist_ok=True)

# vocab for clip
vocab_url = "https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt"
vocab_dest = outpath / vocab_url.split("/")[-1]
if not vocab_dest.exists():
    print("downloading clip vocab")
    import requests
    with requests.get(vocab_url, stream=True) as r:
        assert r.status_code == 200, f"{vocab_url} failed to download. please copy it to {vocab_dest} manually."
        with vocab_dest.open('wb') as vf:
            for c in r.iter_content(chunk_size=8192):
                vf.write(c)
        print("downloaded clip vocab")

# model weights
for k in ckpt["state_dict"]:
    if "first_stage_model.encoder" in k: continue
    ckpt["state_dict"][k].numpy().astype('float16').tofile(outpath / (k + ".bin"))
    print("exporting state_dict", k, end="\r")
print("\nexporting other stuff...")

# other stuff
th.exp(-th.log(th.tensor([10000])) * th.arange(0, 160) / 160).numpy().tofile(outpath / "temb_coefficients_fp32.bin")
np.triu(np.ones((1,1,77,77), dtype=np.float16) * -65500.0, k=1).astype(np.float16).tofile(outpath / "causal_mask.bin")
np.array([0.14013671875, 0.0711669921875, -0.03271484375, -0.11407470703125, 0.126220703125, 0.10101318359375, 0.034515380859375, -0.1383056640625, 0.126220703125, 0.07733154296875, 0.042633056640625, -0.177978515625]).astype(np.float16).tofile(outpath / "aux_output_conv.weight.bin")
np.array([0.423828125, 0.471923828125, 0.473876953125]).astype(np.float16).tofile(outpath / "aux_output_conv.bias.bin")
print(f"Done!")
