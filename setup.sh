#!/usr/bin/env bash
# Fetch-check for the vendored quad engines (gitignored — not in the clone).
# Installs Python deps, then verifies the engines QUAD mode needs are present.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> Installing Python dependencies…"
python3 -m pip install -q -r requirements.txt

QW="_quadwild/quadwild"
NC="_neurcross/quad_mesh/train_quad_mesh.py"
ok=1

echo ""
if [ -x "$QW" ] && [ -f "_quadwild/config/main_config/flow.txt" ]; then
  echo ">> QuadWild-BiMDF (QUAD mode) ......... present"
else
  ok=0
  cat <<'MSG'
>> QuadWild-BiMDF (QUAD mode) ......... MISSING
   QUAD mode needs the QuadWild-BiMDF prebuilt Linux binary. It is gitignored
   (large, GPL-3, third-party). Get it from:
     https://github.com/cgg-bern/quadwild-bimdf  (releases)
   and place it so that these exist:
     _quadwild/quadwild
     _quadwild/quad_from_patches
     _quadwild/config/...   (prep_config/, main_config/flow.txt, …)
   TRIS mode and the in-house quad fallbacks work without it.
MSG
fi

echo ""
if [ -f "$NC" ]; then
  echo ">> NeurCross (neural engine, optional) ... present"
else
  cat <<'MSG'
>> NeurCross (neural engine, optional) ... not installed
   Only needed for the NeurCross engine (organic shapes, needs a CUDA GPU).
   Clone into _neurcross/ and preinstall a CUDA torch:
     git clone https://github.com/QiujieDong/NeurCross _neurcross
   Without it, the NeurCross engine option simply falls back to QuadWild.
MSG
fi

echo ""
if [ "$ok" = "1" ]; then
  echo ">> Ready. Run ./run.sh and open http://127.0.0.1:8000"
else
  echo ">> Set up the engine(s) above, then run ./run.sh"
fi
