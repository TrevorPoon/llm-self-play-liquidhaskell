# -*- coding: utf-8 -*-
"""
Validate a noisy collection of Haskell snippets.

 * keeps only definitions that:
     – have a top-level binding
     – need at least one input (i.e. their first good type signature contains ->)
     – type-check with GHC

 * writes the passing rows back as an HF Dataset + .jsonl
"""
import os
import sys
import re
import json
import textwrap
import logging
import argparse
import subprocess
import tempfile
from typing import List, Tuple, Optional

from datasets import load_from_disk, Dataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# regex helpers
TYPE_SIG_RE = re.compile(
    r"""^          # start of line
        ([()\w\+\*/=<>&|:'\[\]$~!@#%^?-]+)  # name or operator
        \s*::\s*                            # ::
        (.+)$                               # the full type (rest of line)
    """, re.VERBOSE | re.MULTILINE)

OPERATOR_NAME_RE = re.compile(r"^\([^)]*\)$")   # e.g. (+), (>>>)

# -----------------------------------------------------------------------------


def strip_comments(src: str) -> str:
    """Drop {- -}, --, and {-# #-} pragmas, keep line count."""
    # pragmas / block comments
    src = re.sub(r"{-\#[\s\S]*?\#-}", "", src)
    src = re.sub(r"{-[\s\S]*?-}", "", src)
    # line comments
    return re.sub(r"--.*", "", src)


def find_type_sigs(src: str) -> List[Tuple[str, str]]:
    """Return a list of (name, typeSignature) in order of appearance."""
    cleaned = strip_comments(src)
    return TYPE_SIG_RE.findall(cleaned)


def remove_leading_context(ty: str) -> str:
    """Drop `(Num a, Show b) =>` part from a type."""
    return re.sub(r"^\s*\(.*?\)\s*=>\s*", "", ty, count=1).strip()


def needs_input(ty: str) -> bool:
    """True if the core type (after stripping context) contains a top-level ->"""
    ty = remove_leading_context(ty)
    depth = 0
    for ch in ty:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == "-" and depth == 0:      # we will see '-' of '->'
            return True
    return False


def has_function_binding(src: str, name: str) -> bool:
    """
    Look for a top-level equation for `name`.
    Accepts ordinary identifiers and operator names in (…).
    """
    cleaned = strip_comments(src)
    if OPERATOR_NAME_RE.fullmatch(name):
        # operators are defined like (++) xs ys = …
        pat = rf"^{re.escape(name)}\s+.*="
    else:
        pat = rf"^{re.escape(name)}\b.*="
    return re.search(pat, cleaned, re.MULTILINE) is not None


# ──────────────────────────────────────────────────────────────────────────────
# fast compile check
class CodeChecker:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.tmp_dir = tempfile.mkdtemp(prefix="hs_chk_")

    def _wrap_module(self, src: str) -> str:
        """Add Prelude imports so we can always run GHC -fno-code."""
        imports = textwrap.dedent("""
            import Prelude
            import Data.List
            import Data.Char
            import Data.Maybe
        """).strip()
        return f"{imports}\n\n{src}"

    def typecheck(self, src: str) -> bool:
        mod = self._wrap_module(src)
        with tempfile.NamedTemporaryFile("w", suffix=".hs",
                                         dir=self.tmp_dir, delete=False) as f:
            f.write(mod)
            tmp_path = f.name
        try:
            res = subprocess.run(
                ["ghc", "-fno-code", "-v0", tmp_path],
                capture_output=True, text=True, timeout=self.timeout
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("GHC timed out")
            return False
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
def validate_dataset(args):
    logger.info(f"Loading dataset from {args.dataset_path}")
    try:
        ds = load_from_disk(args.dataset_path)
    except Exception as e:
        logger.error(f"Cannot load dataset: {e}")
        return

    checker = CodeChecker(timeout=args.timeout)

    good_rows = []
    counts = {
        "total": len(ds),
        "with_sig": 0,
        "with_binding": 0,
        "needs_input": 0,
        "typechecked": 0
    }

    for row in tqdm(ds, desc="Validating"):
        code = row.get("code", "")
        if not code.strip():
            continue

        sigs = find_type_sigs(code)
        if not sigs:
            continue
        counts["with_sig"] += 1

        # use first signature that has both binding and needs input
        accepted = False
        for name, ty in sigs:
            if not needs_input(ty):
                continue
            if not has_function_binding(code, name):
                continue

            counts["with_binding"] += 1
            counts["needs_input"] += 1

            if checker.typecheck(code):
                counts["typechecked"] += 1
                good_rows.append(row)
            accepted = True
            break

        # if no sig passed the filters we just move on
        if not accepted:
            continue

    # summary
    logger.info("─── summary ───────────────────────────")
    for k, v in counts.items():
        logger.info(f"{k.replace('_', ' ')}: {v}")
    logger.info(f"accepted rows: {len(good_rows)}")

    if not good_rows:
        logger.warning("Nothing to save.")
        return

    out_dir = os.path.join(args.output_dir, "validated_haskell_dataset")
    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_list(good_rows).save_to_disk(out_dir)
    logger.info(f"Saved HuggingFace dataset → {out_dir}")

    jsonl_path = os.path.join(args.output_dir, "validated_haskell_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in good_rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved JSONL → {jsonl_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))

    parser = argparse.ArgumentParser("Validate Haskell programs")
    parser.add_argument("--dataset_path",
                        default=os.path.join(parent_dir, "data", "sorted_blastwind_haskell_dataset"),
                        help="Path to HuggingFace dataset on disk")
    parser.add_argument("--output_dir",
                        default=os.path.join(parent_dir, "data"),
                        help="Directory for outputs")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="GHC timeout per file (seconds)")

    args = parser.parse_args()
    validate_dataset(args)


if __name__ == "__main__":
    main()
