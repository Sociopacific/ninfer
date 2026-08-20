"""Rebuild a Qwen3.8-27B NVFP4 artifact around a DFlash2 drafter.

The published artifact already carries every text, vision and draft-head
object.  Re-deriving those from the upstream checkpoints costs an hour and
seventy gigabytes of downloads, so this tool copies them verbatim and only
encodes the ``dflash/*`` objects from a DFlash2 drafter checkpoint.  Objects
the current inventory no longer declares -- the MTP head -- are left behind.

    python3 -m tools.convert.qwen3_8_27b.splice_dflash \
        --artifact models/nvfp4/qwen3_8_27b_nvfp4.ninfer \
        --dflash-model models/dflash2-qwen38-27b \
        --out models/dflash2/qwen3_8_27b_nvfp4.ninfer
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe
from .convert_nvfp4 import open_source


OUTPUT_BASENAME = "qwen3_8_27b_nvfp4.ninfer"


def splice(
    artifact_path: str | Path,
    dflash_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"artifact basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    inventory.validate_inventory()
    recipe.validate_recipe()

    resolved_device = torch.device(device)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Artifact.open(artifact_path) as source:
        carried = {
            spec.name
            for spec in inventory.OBJECT_SPECS
            if spec.name not in recipe.DFLASH_RECIPES_BY_NAME
        }
        missing = sorted(name for name in carried if not _has(source, name))
        if missing:
            raise ValueError(
                f"source artifact lacks {len(missing)} carried objects, "
                f"first={missing[0]}"
            )
        resources = {
            spec.name: bytes(source.payload(spec.name))
            for spec in inventory.OBJECT_SPECS
            if isinstance(spec, inventory.ResourceSpec)
        }
        plan = family_conversion.build_object_plan(
            inventory.OBJECT_SPECS, resources
        )
        dropped = sorted(
            obj.name
            for obj in source.objects
            if obj.name not in {spec.name for spec in inventory.OBJECT_SPECS}
        )
        print(
            f"carrying {len(carried)} objects, encoding "
            f"{len(recipe.DFLASH_RECIPES_BY_NAME)} dflash objects, "
            f"dropping {len(dropped)}",
            flush=True,
        )
        if dropped:
            print("dropped: " + ", ".join(dropped), flush=True)

        with open_source(dflash_dir) as dflash_reader, ArtifactWriter(
            output, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID), plan.specs
        ) as writer:
            total = len(inventory.OBJECT_SPECS)
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if spec.name in recipe.DFLASH_RECIPES_BY_NAME:
                    tensor = recipe.materialize_dflash(spec.name, dflash_reader)
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{spec.name}: materialized shape "
                            f"{tuple(tensor.shape)} != {spec.shape}"
                        )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                    print(f"[{index}/{total}] {spec.name} (encoded)", flush=True)
                else:
                    payload = source.payload(spec.name)
                writer.write(spec.name, payload)
                del payload

    elapsed = time.perf_counter() - started
    print(
        f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s -> {output}",
        flush=True,
    )
    return output


def _has(source: Artifact, name: str) -> bool:
    try:
        source.find(name)
    except Exception:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dflash-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    splice(
        arguments.artifact,
        arguments.dflash_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
