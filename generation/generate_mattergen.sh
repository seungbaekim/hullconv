#!/usr/bin/env bash
# Generate structures with MatterGen conditioned on a chemical system and on an
# energy above the hull of 0.03 eV/atom, using the released fine-tuned
# checkpoint chemical_system_energy_above_hull and a classifier-free guidance
# factor of 2.0 (all other sampling settings at their defaults).
#
# Usage: generation/generate_mattergen.sh <chemical-system> <output-dir> [batch-size] [num-batches]
# Example (10,000 structures): generation/generate_mattergen.sh Li-Ru-O runs/LRO 100 100
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <chemical-system> <output-dir> [batch-size] [num-batches]" >&2
    exit 1
fi

CHEMICAL_SYSTEM="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-100}"
NUM_BATCHES="${4:-100}"
EHULL_TARGET=0.03     # eV/atom
GUIDANCE_FACTOR=2.0

mattergen-generate "$OUTPUT_DIR" \
    --pretrained-name=chemical_system_energy_above_hull \
    --batch_size="$BATCH_SIZE" \
    --num_batches="$NUM_BATCHES" \
    --properties_to_condition_on="{'energy_above_hull': ${EHULL_TARGET}, 'chemical_system': '${CHEMICAL_SYSTEM}'}" \
    --diffusion_guidance_factor="$GUIDANCE_FACTOR"
