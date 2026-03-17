#!/usr/bin/env python3
"""Pre-build static validation for this Android device/vendor tree.

Designed to work in minimal environments (no sudo, no ripgrep).
It performs best-effort checks and can apply only conservative autofixes.

Usage:
  tools/prebuild_validate.py --root /path/to/tree
  tools/prebuild_validate.py --root . --fix
    tools/prebuild_validate.py --root . --elf
    tools/prebuild_validate.py --root . --elf --elf-max 200

Exit codes:
  0: no issues found
  1: issues found (or fixes applied)
  2: internal error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


REPOS_DEFAULT = [
    "android_device_samsung_a52sxq",
    "android_device_samsung_sm7325_common",
    "android_vendor_samsung_a52sxq",
    "android_vendor_samsung_sm7325-common",
    "android_kernel_samsung_sm7325",
]


CORE_VENDOR_BIN_ASSUMED = {
    # Common platform-provided tools often present on /vendor but not defined here.
    "sh",
    "toybox",
    "toolbox",
}


@dataclass(frozen=True)
class Issue:
    repo: str
    category: str
    severity: str  # info|warn|error
    file: Path | None
    message: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_files(root: Path, rel_repo: str, patterns: list[str]) -> Iterator[Path]:
    base = root / rel_repo
    if not base.exists():
        return
    for pattern in patterns:
        yield from base.rglob(pattern)


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#")


def _parse_init_imports_and_services(rc_path: Path) -> tuple[list[str], list[str]]:
    imports: list[str] = []
    exec_paths: list[str] = []

    # Keep this intentionally simple; init rc syntax is not fully parsed.
    import_re = re.compile(r"^\s*import\s+(\S+)\s*$")
    service_re = re.compile(r"^\s*service\s+\S+\s+(\S+)(?:\s+.*)?$")
    exec_re = re.compile(r"^\s*exec(?:_background)?\s+(?:[^/\s]\S+\s+)*(/\S+)(?:\s+.*)?$")

    for raw in _read_text(rc_path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = import_re.match(raw)
        if m:
            imports.append(m.group(1))
            continue

        m = service_re.match(raw)
        if m:
            exec_paths.append(m.group(1))
            continue

        m = exec_re.match(raw)
        if m:
            exec_paths.append(m.group(1))
            continue

    return imports, exec_paths


@dataclass(frozen=True)
class InitServiceDef:
    name: str
    header_norm: str
    body_norm: tuple[str, ...]
    has_override: bool


def _normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())


def _parse_init_service_defs(rc_path: Path) -> list[InitServiceDef]:
    """Parse init rc service stanzas (best-effort).

    Recognizes the `override` option as a signal that duplicate service names are intentional.
    """

    service_hdr_re = re.compile(r"^\s*service\s+(\S+)\s+(\S+)(?:\s+.*)?$")
    out: list[InitServiceDef] = []

    cur_name: str | None = None
    cur_header_norm: str | None = None
    cur_body: list[str] = []
    cur_override = False

    def flush() -> None:
        nonlocal cur_name, cur_header_norm, cur_body, cur_override
        if cur_name and cur_header_norm:
            body_norm = tuple(_normalize_ws(x) for x in cur_body if _normalize_ws(x))
            out.append(
                InitServiceDef(
                    name=cur_name,
                    header_norm=cur_header_norm,
                    body_norm=body_norm,
                    has_override=cur_override,
                )
            )
        cur_name = None
        cur_header_norm = None
        cur_body = []
        cur_override = False

    for raw in _read_text(rc_path).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        m = service_hdr_re.match(raw)
        if m:
            flush()
            cur_name = m.group(1)
            cur_header_norm = _normalize_ws(raw)
            continue

        # Stanza body lines are indented. Any non-indented directive ends the stanza.
        if cur_name:
            if raw[:1].isspace():
                stmt = _normalize_ws(raw)
                if stmt == "override":
                    cur_override = True
                cur_body.append(raw)
            else:
                flush()

    flush()
    return out


def _parse_init_service_names(rc_path: Path) -> list[str]:
    return [d.name for d in _parse_init_service_defs(rc_path)]


def _normalize_runtime_path(p: str) -> str:
    # /system/vendor is typically a symlink to /vendor; treat it as an alias to reduce false positives.
    if p.startswith("/system/vendor/"):
        return "/vendor/" + p[len("/system/vendor/") :]
    return p


def _partition_path_candidates(root: Path, absolute_path: str) -> list[Path]:
    """Map a runtime absolute partition path to candidate source-tree paths."""
    p = _normalize_runtime_path(absolute_path)

    rel = p.lstrip("/")
    candidates: list[Path] = []

    # Known proprietary tree partitions in these repos.
    for repo in ("android_vendor_samsung_sm7325-common", "android_vendor_samsung_a52sxq"):
        base = root / repo / "proprietary"
        for part in ("vendor", "product", "system_ext"):
            if rel.startswith(part + "/"):
                # Paths like /vendor/... should map under proprietary/vendor/...
                candidates.append(base / rel)
                break
        else:
            # Also allow mapping /vendor/... into proprietary/vendor/... (common case).
            if rel.startswith("vendor/"):
                candidates.append(base / rel)
            elif rel.startswith("product/"):
                candidates.append(base / rel)
            elif rel.startswith("system_ext/"):
                candidates.append(base / rel)

    return candidates


@dataclass
class BuildIndex:
    # Absolute paths expected to exist at runtime.
    vendor_etc_paths: set[str]
    vendor_bin_paths: set[str]
    modules: set[str]
    # Map of vendor mk paths copied into partitions.
    vendor_mk_copy_srcs: set[Path]
    vendor_mk_copy_dests: set[str]


def _parse_android_bp_modules(bp: Path) -> tuple[set[str], set[str]]:
    """Very small subset parser for prebuilt_etc and sh_binary modules.

    Returns:
      (vendor_etc_paths, vendor_bin_paths)
    """

    vendor_etc: set[str] = set()
    vendor_bin: set[str] = set()

    current_kind: str | None = None
    props: dict[str, str] = {}

    kind_re = re.compile(r"^\s*(prebuilt_etc|sh_binary)\s*\{\s*$")
    prop_re = re.compile(r"^\s*(name|src|sub_dir|filename)\s*:\s*\"([^\"]+)\"\s*,?\s*$")
    end_re = re.compile(r"^\s*\}\s*,?\s*$")

    for raw in _read_text(bp).splitlines():
        m = kind_re.match(raw)
        if m:
            current_kind = m.group(1)
            props = {}
            continue

        if current_kind:
            m = prop_re.match(raw)
            if m:
                props[m.group(1)] = m.group(2)
                continue
            if end_re.match(raw):
                if current_kind == "prebuilt_etc":
                    src = props.get("src")
                    sub_dir = props.get("sub_dir", "")
                    filename = props.get("filename")
                    if src:
                        out_name = filename or Path(src).name
                        # Assume vendor placement for soc_specific modules in this tree.
                        sub = sub_dir.strip("/")
                        rel = "etc"
                        if sub:
                            rel = f"{rel}/{sub}"
                        vendor_etc.add(f"/vendor/{rel}/{out_name}")
                elif current_kind == "sh_binary":
                    name = props.get("name")
                    if name:
                        vendor_bin.add(f"/vendor/bin/{name}")

                current_kind = None
                props = {}

    return vendor_etc, vendor_bin


def _parse_product_packages_from_mk(mk: Path) -> set[str]:
    modules: set[str] = set()
    in_block = False
    for raw in _read_text(mk).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if not in_block:
            if re.match(r"^PRODUCT_PACKAGES\s*\+=", line):
                in_block = True
                rhs = line.split("+=", 1)[1]
                tokens = [t for t in rhs.replace("\\", " ").split() if t and t != "\\"]
                modules.update(tokens)
                if not line.endswith("\\"):
                    in_block = False
            continue

        # inside continuation
        tokens = [t for t in line.replace("\\", " ").split() if t and t != "\\"]
        modules.update(tokens)
        if not line.endswith("\\"):
            in_block = False

    return modules


def _parse_product_copy_files_from_mk(mk: Path) -> tuple[set[str], set[Path]]:
    """Return (dest_paths, src_paths) from PRODUCT_COPY_FILES entries."""
    dests: set[str] = set()
    srcs: set[Path] = set()

    in_block = False
    for raw in _read_text(mk).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if not in_block:
            if re.match(r"^PRODUCT_COPY_FILES\s*\+=", line):
                in_block = True
                rhs = line.split("+=", 1)[1]
                parts = [p.strip() for p in rhs.split() if p.strip() and p.strip() != "\\"]
                for p in parts:
                    if ":" not in p:
                        continue
                    src, dest = p.split(":", 1)
                    if src:
                        srcs.add(Path(src))
                    if dest:
                        dests.add(dest)
                if not line.endswith("\\"):
                    in_block = False
            continue

        parts = [p.strip() for p in line.split() if p.strip() and p.strip() != "\\"]
        for p in parts:
            if ":" not in p:
                continue
            src, dest = p.split(":", 1)
            if src:
                srcs.add(Path(src))
            if dest:
                dests.add(dest)
        if not line.endswith("\\"):
            in_block = False

    return dests, srcs


def _vendor_namespace_for_repo(repo: str) -> str | None:
    if repo == "android_vendor_samsung_a52sxq":
        return "vendor/samsung/a52sxq"
    if repo == "android_vendor_samsung_sm7325-common":
        return "vendor/samsung/sm7325-common"
    return None


def _aosp_path_to_local_repo_prefix(path_str: str) -> str | None:
    # Map canonical AOSP paths to this workspace's repo naming.
    mappings = {
        "vendor/samsung/a52sxq": "android_vendor_samsung_a52sxq",
        "vendor/samsung/sm7325-common": "android_vendor_samsung_sm7325-common",
        "vendor/samsung/sm7325_common": "android_vendor_samsung_sm7325-common",
        "device/samsung/a52sxq": "android_device_samsung_a52sxq",
        "device/samsung/sm7325_common": "android_device_samsung_sm7325_common",
        "device/samsung/sm7325-common": "android_device_samsung_sm7325_common",
    }
    for aosp_prefix, local_prefix in mappings.items():
        if path_str == aosp_prefix or path_str.startswith(aosp_prefix + "/"):
            suffix = path_str[len(aosp_prefix) :].lstrip("/")
            return f"{local_prefix}/{suffix}" if suffix else local_prefix
    return None


def _resolve_in_tree_path(root: Path, path_str: str, relative_to: Path | None = None) -> Path | None:
    """Resolve an AOSP-ish path string into a file path in this workspace.

    Returns a Path if it can be resolved deterministically, else None.
    """
    s = path_str.strip().strip("\"")
    if not s or "$(" in s or "`" in s:
        return None

    # Handle $(LOCAL_PATH)/... for the current makefile.
    if relative_to and s.startswith("$(LOCAL_PATH)/"):
        return relative_to.parent / s[len("$(LOCAL_PATH)/") :]

    mapped = _aosp_path_to_local_repo_prefix(s)
    if mapped:
        return root / mapped

    return root / s


def check_makefile_includes(root: Path, repos: list[str]) -> list[Issue]:
    issues: list[Issue] = []

    # Very small include/inherit-product validator.
    inherit_re = re.compile(r"\$\(call\s+inherit-product(?:-if-exists)?,\s*([^\)]+)\)")
    include_re = re.compile(r"^\s*-?include\s+(.+?)\s*$")

    external_prefixes = (
        "hardware/",
        "frameworks/",
        "system/",
        "build/",
        "packages/",
        "device/qcom",
        "device/lineage",
        "vendor/qcom",
        "vendor/lineage",
        "vendor/sony",
    )

    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        for mk in base.rglob("*.mk"):
            text = _read_text(mk)
            for idx, raw in enumerate(text.splitlines(), start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # include / -include
                m = include_re.match(raw)
                if m:
                    target = m.group(1).strip()
                    resolved = _resolve_in_tree_path(root, target, relative_to=mk)
                    if resolved is None:
                        continue
                    if not resolved.exists():
                        sev = "warn"
                        if target.startswith(external_prefixes):
                            sev = "warn"
                        elif target.startswith(("device/", "vendor/")):
                            sev = "error"
                        issues.append(
                            Issue(repo, "build", sev, mk, f"Missing include at line {idx}: {target} (resolved {resolved})")
                        )
                    continue

                # inherit-product
                m = inherit_re.search(raw)
                if m:
                    target = m.group(1).strip()
                    resolved = _resolve_in_tree_path(root, target, relative_to=mk)
                    if resolved is None:
                        continue
                    if not resolved.exists():
                        sev = "warn"
                        if target.startswith(external_prefixes):
                            sev = "warn"
                        elif target.startswith(("device/", "vendor/")):
                            sev = "error"
                        issues.append(
                            Issue(repo, "build", sev, mk, f"Missing inherit-product target at line {idx}: {target} (resolved {resolved})")
                        )

    return issues


def _resolve_vendor_mk_src(root: Path, repo_base: Path, repo: str, src: Path) -> Path:
    """Resolve a PRODUCT_COPY_FILES source path from a *-vendor.mk.

    In a normal AOSP tree, vendor repos are checked out under vendor/samsung/...,
    but in this workspace they live under android_vendor_samsung_*.
    """

    if src.is_absolute():
        return src

    src_str = str(src).replace(os.sep, "/")
    ns = _vendor_namespace_for_repo(repo)
    if ns and src_str.startswith(ns + "/"):
        rel = src_str[len(ns) + 1 :]
        return repo_base / rel

    # Best-effort fallback: resolve relative to the root.
    return root / src


def build_index(root: Path, repos: list[str]) -> BuildIndex:
    vendor_etc_paths: set[str] = set()
    vendor_bin_paths: set[str] = set()
    modules: set[str] = set()
    copy_srcs: set[Path] = set()
    copy_dests: set[str] = set()

    # Index device trees for init scripts and product packages.
    for repo in repos:
        base = root / repo
        if not base.exists():
            continue

        # Soong modules
        for bp in base.rglob("Android.bp"):
            ve, vb = _parse_android_bp_modules(bp)
            vendor_etc_paths |= ve
            vendor_bin_paths |= vb

        # Make modules/packages and copy files.
        for mk in base.rglob("*.mk"):
            modules |= _parse_product_packages_from_mk(mk)
            dests, srcs = _parse_product_copy_files_from_mk(mk)
            copy_dests |= dests
            copy_srcs |= {root / s for s in srcs if not s.is_absolute()}

    return BuildIndex(
        vendor_etc_paths=vendor_etc_paths,
        vendor_bin_paths=vendor_bin_paths,
        modules=modules,
        vendor_mk_copy_srcs=copy_srcs,
        vendor_mk_copy_dests=copy_dests,
    )


def _vendor_path_candidates(root: Path, absolute_path: str) -> list[Path]:
    """Backward-compat shim for old callers: only maps /vendor/... and /system/vendor/..."""
    p = _normalize_runtime_path(absolute_path)
    if not p.startswith("/vendor/"):
        return []
    return _partition_path_candidates(root, p)


def _exists_any(paths: Iterable[Path]) -> bool:
    return any(p.exists() for p in paths)


def check_init_imports_and_services(root: Path, repos: list[str], index: BuildIndex) -> list[Issue]:
    issues: list[Issue] = []

    init_patterns = ["*.rc"]
    repo_to_rc_roots = {
        "android_device_samsung_a52sxq": ["init"],
        "android_device_samsung_sm7325_common": ["init"],
        "android_vendor_samsung_sm7325-common": ["proprietary/vendor/etc/init"],
        "android_vendor_samsung_a52sxq": ["proprietary/vendor/etc/init"],
    }

    # Collect all init rc files.
    rc_files: list[tuple[str, Path]] = []
    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        for sub in repo_to_rc_roots.get(repo, []):
            subdir = base / sub
            if not subdir.exists():
                continue
            for rc in subdir.rglob("*.rc"):
                rc_files.append((repo, rc))

    # Parse and validate.
    service_to_files: dict[str, list[Path]] = {}
    for repo, rc in rc_files:
        try:
            imports, exec_paths = _parse_init_imports_and_services(rc)
            service_names = _parse_init_service_names(rc)
        except Exception as e:  # pragma: no cover
            issues.append(Issue(repo, "init", "error", rc, f"Failed to parse: {e}"))
            continue

        for sn in service_names:
            service_to_files.setdefault(sn, []).append(rc)

        for imp in imports:
            imp = _normalize_runtime_path(imp)
            if "${" in imp:
                issues.append(
                    Issue(
                        repo,
                        "init",
                        "info",
                        rc,
                        f"Dynamic import not statically verifiable: {imp}",
                    )
                )
                continue

            if imp.startswith("/vendor/"):
                if imp in index.vendor_etc_paths:
                    continue
                candidates = _vendor_path_candidates(root, imp)
                if _exists_any(candidates):
                    continue

                # Best-effort: if the file exists somewhere in the tree but isn't indexed
                # as installed to vendor/etc, call it out as a packaging risk.
                basename = Path(imp).name
                anywhere = list(root.rglob(basename))
                if anywhere:
                    issues.append(
                        Issue(
                            repo,
                            "init",
                            "warn",
                            rc,
                            f"Import {imp} not found under vendor proprietary and not indexed as prebuilt_etc; found similarly-named file(s): {', '.join(str(p) for p in anywhere[:3])}",
                        )
                    )
                else:
                    issues.append(
                        Issue(
                            repo,
                            "init",
                            "warn",
                            rc,
                            f"Import target not found in this tree: {imp} (may come from external QCOM repos)",
                        )
                    )
            elif imp.startswith("/"):
                issues.append(
                    Issue(
                        repo,
                        "init",
                        "warn",
                        rc,
                        f"Import points outside /vendor (not validated here): {imp}",
                    )
                )

        for ep in exec_paths:
            ep = _normalize_runtime_path(ep)
            if "${" in ep:
                issues.append(Issue(repo, "init", "info", rc, f"Dynamic exec path: {ep}"))
                continue

            if ep.startswith("/vendor/"):
                # If this path is known to be produced by Soong in our device trees.
                if ep in index.vendor_bin_paths:
                    continue

                # Common init paths reference /vendor/bin or /vendor/bin/hw.
                if ep.startswith("/vendor/bin/hw/"):
                    mod = ep.rsplit("/", 1)[1]
                    if mod in index.modules:
                        continue
                if ep.startswith("/vendor/bin/"):
                    mod = ep.rsplit("/", 1)[1]
                    if mod in index.modules:
                        continue
                    if mod in CORE_VENDOR_BIN_ASSUMED:
                        issues.append(Issue(repo, "init", "info", rc, f"Assuming core vendor tool exists at runtime: {ep}"))
                        continue

                candidates = _vendor_path_candidates(root, ep)
                if not _exists_any(candidates):
                    issues.append(
                        Issue(
                            repo,
                            "init",
                            "warn",
                            rc,
                            f"Service/exec references {ep} which is not found in vendor proprietary and not indexed as built module (may be external or missing)",
                        )
                    )

    return issues


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _readelf_needed(path: Path) -> tuple[list[str], str | None]:
    """Return (NEEDED libs, error message)."""
    if not _is_elf(path):
        return ([], None)

    try:
        p = subprocess.run(
            ["readelf", "-d", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return ([], f"readelf failed: {e}")

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        return ([], err or f"readelf exited {p.returncode}")

    needed: list[str] = []
    for line in p.stdout.splitlines():
        if "(NEEDED)" not in line:
            continue
        m = re.search(r"\[(.+?)\]", line)
        if m:
            needed.append(m.group(1))
    return (needed, None)


def check_elf_dependency_closure(root: Path, repos: list[str], max_binaries: int) -> list[Issue]:
    """Check that ELF binaries referenced by init can find their DT_NEEDED libs in-tree."""
    issues: list[Issue] = []

    # Many DT_NEEDED libs for /vendor binaries come from the platform/VNDK and won't exist in
    # proprietary blob trees. Keep this allowlist conservative to reduce noise.
    platform_libs = {
        "libc.so",
        "libm.so",
        "libdl.so",
        "liblog.so",
        "libcutils.so",
        "libutils.so",
        "libbase.so",
        "libc++.so",
        "libbinder.so",
        "libhidlbase.so",
        "libhidltransport.so",
        "libhardware.so",
        "libz.so",
        "liblz4.so",
        "libcrypto.so",
        "libssl.so",
        "libpthread.so",
    }

    # Index libs by basename across proprietary trees.
    lib_index: dict[str, list[Path]] = {}
    for repo in ("android_vendor_samsung_sm7325-common", "android_vendor_samsung_a52sxq"):
        base = root / repo / "proprietary"
        if not base.exists():
            continue
        for so in base.rglob("*.so"):
            lib_index.setdefault(so.name, []).append(so)

    # Collect init exec targets.
    exec_targets: list[tuple[str, Path, str]] = []
    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        for rc in base.rglob("*.rc"):
            try:
                _, execs = _parse_init_imports_and_services(rc)
            except Exception:
                continue
            for ep in execs:
                ep_n = _normalize_runtime_path(ep)
                if ep_n.startswith("/vendor/"):
                    exec_targets.append((repo, rc, ep_n))

    # Resolve to actual binaries and run readelf.
    checked = 0
    seen_bin: set[Path] = set()
    for repo, rc, ep in exec_targets:
        if checked >= max_binaries:
            break

        candidates = _partition_path_candidates(root, ep)
        bin_path = next((c for c in candidates if c.exists()), None)
        if not bin_path:
            continue
        if bin_path in seen_bin:
            continue
        seen_bin.add(bin_path)
        checked += 1

        needed, err = _readelf_needed(bin_path)
        if err:
            issues.append(Issue(repo, "elf", "warn", rc, f"Failed to read DT_NEEDED for {ep} ({bin_path}): {err}"))
            continue

        missing = [n for n in needed if n not in platform_libs and n not in lib_index]
        if missing:
            issues.append(
                Issue(
                    repo,
                    "elf",
                    "warn",
                    rc,
                    f"{ep} DT_NEEDED missing in proprietary trees: {', '.join(sorted(set(missing)))} (binary {bin_path})",
                )
            )

    return issues


def _remove_exact_duplicate_service_stanza(rc_path: Path, service_name: str, signature: tuple[str, tuple[str, ...]]) -> int:
    """Remove service stanzas matching (header_norm, body_norm). Returns removed stanza count."""

    service_hdr_re = re.compile(r"^\s*service\s+(\S+)\s+\S+(?:\s+.*)?$")
    lines = _read_text(rc_path).splitlines(keepends=True)

    out: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        raw = lines[i]
        m = service_hdr_re.match(raw)
        if not m or m.group(1) != service_name:
            out.append(raw)
            i += 1
            continue

        # Capture stanza.
        hdr_norm = _normalize_ws(raw)
        body: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j][:1].isspace():
            body.append(lines[j])
            j += 1
        body_norm = tuple(_normalize_ws(x) for x in body if _normalize_ws(x))

        if (hdr_norm, body_norm) == signature:
            removed += 1
            i = j
            continue

        # Not an exact match; keep it.
        out.append(raw)
        out.extend(body)
        i = j

    if removed:
        rc_path.write_text("".join(out), encoding="utf-8")
    return removed


def check_init_duplicate_services(root: Path, repos: list[str], fix: bool) -> tuple[list[Issue], list[Issue]]:
    issues: list[Issue] = []
    fixes: list[Issue] = []

    rc_files: list[Path] = []
    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        rc_files.extend(base.rglob("*.rc"))

    service_to_defs: dict[str, list[tuple[Path, InitServiceDef]]] = {}
    for rc in rc_files:
        try:
            for d in _parse_init_service_defs(rc):
                service_to_defs.setdefault(d.name, []).append((rc, d))
        except Exception:
            continue

    for sn, defs in sorted(service_to_defs.items()):
        if len(defs) <= 1:
            continue

        files = sorted({str(p) for p, _ in defs})
        has_override = any(d.has_override for _, d in defs)
        if has_override:
            issues.append(
                Issue(
                    repo="(multi)",
                    category="init",
                    severity="info",
                    file=None,
                    message=f"Service {sn!r} is defined in multiple rc files but uses 'override' (likely intentional): {', '.join(files[:5])}",
                )
            )
            continue

        # If all stanza signatures are identical, we can optionally auto-fix by removing duplicates.
        sigs = {(d.header_norm, d.body_norm) for _, d in defs}
        if fix and len(sigs) == 1:
            signature = next(iter(sigs))
            # Keep the lexicographically earliest rc to match typical init directory load ordering.
            keep_rc = sorted({p for p, _ in defs}, key=lambda p: str(p))[:1][0]
            removed_total = 0
            for rc, _d in defs:
                if rc == keep_rc:
                    continue
                removed = _remove_exact_duplicate_service_stanza(rc, sn, signature)
                removed_total += removed
            if removed_total:
                fixes.append(Issue("(multi)", "init", "info", keep_rc, f"Removed {removed_total} exact-duplicate service stanza(s) for {sn!r}; kept definition in {keep_rc}"))
                continue

        issues.append(
            Issue(
                repo="(multi)",
                category="init",
                severity="warn",
                file=None,
                message=f"Duplicate init service name {sn!r} defined in: {', '.join(files[:5])}",
            )
        )

    return issues, fixes


def _parse_proprietary_file_line(line: str) -> str | None:
    """Return destination path (as found in proprietary/) or None."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    # Drop any pinned hashes or arguments.
    s = s.split("|", 1)[0].strip()
    s = s.split(";", 1)[0].strip()

    # Lines may be prefixed with '-' to remove from inherited lists.
    if s.startswith("-"):
        s = s[1:].strip()

    # src:dest form
    if ":" in s:
        _, dest = s.split(":", 1)
        s = dest.strip()

    # Optional marker '?' sometimes used.
    if s.startswith("?"):
        s = s[1:].strip()

    if not s:
        return None

    return s


def check_proprietary_files(root: Path, repos: list[str], index: BuildIndex) -> list[Issue]:
    issues: list[Issue] = []

    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        if "vendor" not in repo:
            continue

        # Two common patterns exist:
        #   1) proprietary-files*.txt (extract_utils)
        #   2) *-vendor.mk generated with PRODUCT_COPY_FILES
        txts = sorted(base.rglob("proprietary-files*.txt"))
        vendor_mks = sorted(base.glob("*-vendor.mk"))

        if txts:
            for txt in txts:
                for idx, raw in enumerate(_read_text(txt).splitlines(), start=1):
                    dest = _parse_proprietary_file_line(raw)
                    if not dest:
                        continue
                    blob = base / "proprietary" / dest
                    if not blob.exists():
                        issues.append(
                            Issue(
                                repo,
                                "proprietary",
                                "error",
                                txt,
                                f"Missing blob for line {idx}: {dest} (expected {blob})",
                            )
                        )
        elif vendor_mks:
            for mk in vendor_mks:
                dests, srcs = _parse_product_copy_files_from_mk(mk)
                for src in sorted(srcs):
                    if src.is_absolute():
                        # Not expected in these generated files.
                        continue
                    src_abs = _resolve_vendor_mk_src(root, base, repo, src)
                    if not src_abs.exists():
                        issues.append(
                            Issue(
                                repo,
                                "proprietary",
                                "error",
                                mk,
                                f"PRODUCT_COPY_FILES source missing: {src} (resolved to {src_abs})",
                            )
                        )
                # Light sanity: ensure vendor.mk destinations look like partition outputs.
                for d in sorted(dests):
                    if "$(TARGET_COPY_OUT_" not in d:
                        issues.append(Issue(repo, "proprietary", "warn", mk, f"Unusual PRODUCT_COPY_FILES dest: {d}"))
        else:
            issues.append(Issue(repo, "proprietary", "warn", None, "No proprietary-files*.txt or *-vendor.mk found"))

        # Note: We intentionally do not try to warn on "blobs present but not referenced" here.
        # That requires correctly resolving which vendor.mk(s) are included for a given build
        # and handling make conditionals; doing it wrong would create lots of false positives.

    return issues


def check_json_files(root: Path, repos: list[str]) -> list[Issue]:
    issues: list[Issue] = []

    # Only check known JSON config names to keep scope tight.
    targets = {"powerhint.json", "task_profiles.json"}
    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if path.name not in targets:
                continue
            try:
                data = json.loads(_read_text(path))
            except Exception as e:
                issues.append(Issue(repo, "json", "error", path, f"Invalid JSON: {e}"))
                continue

            # Heuristic checks (non-fatal): duplicate profile names.
            if path.name == "task_profiles.json" and isinstance(data, dict):
                profiles = data.get("profiles")
                if isinstance(profiles, list):
                    names: list[str] = []
                    for p in profiles:
                        if isinstance(p, dict) and isinstance(p.get("name"), str):
                            names.append(p["name"])
                    dupes = sorted({n for n in names if names.count(n) > 1})
                    if dupes:
                        issues.append(
                            Issue(
                                repo,
                                "json",
                                "warn",
                                path,
                                f"Duplicate task profile names: {', '.join(dupes)}",
                            )
                        )

    return issues


_CONTEXT_RE = re.compile(r"^u:object_r:[A-Za-z0-9_]+:s0(?::c\d+(?:,c\d+)*)?$")


def _dedupe_identical_lines(path: Path) -> tuple[bool, int]:
    """Remove exact duplicate non-comment lines while preserving order."""
    original = _read_text(path).splitlines(keepends=True)
    seen: set[str] = set()
    out: list[str] = []
    removed = 0

    for line in original:
        key = line
        if _is_comment_or_blank(line):
            out.append(line)
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(line)

    if removed:
        path.write_text("".join(out), encoding="utf-8")
        return True, removed
    return False, 0


def check_sepolicy_contexts(root: Path, repos: list[str], fix: bool) -> tuple[list[Issue], list[Issue]]:
    issues: list[Issue] = []
    fixes: list[Issue] = []

    for repo in repos:
        base = root / repo
        if not base.exists():
            continue
        sepol = base / "sepolicy"
        if not sepol.exists():
            continue

        for ctx_name in ("file_contexts", "service_contexts"):
            for ctx in sepol.rglob(ctx_name):
                # Basic format checks and duplicate detection.
                patterns: dict[str, int] = {}
                for idx, raw in enumerate(_read_text(ctx).splitlines(), start=1):
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        issues.append(Issue(repo, "sepolicy", "error", ctx, f"Malformed line {idx}: {raw!r}"))
                        continue
                    pattern = parts[0]
                    context = parts[1]
                    if not _CONTEXT_RE.match(context):
                        issues.append(
                            Issue(
                                repo,
                                "sepolicy",
                                "warn",
                                ctx,
                                f"Suspicious context format at line {idx}: {context}",
                            )
                        )
                    if pattern in patterns:
                        issues.append(
                            Issue(
                                repo,
                                "sepolicy",
                                "warn",
                                ctx,
                                f"Duplicate entry for {pattern!r} at lines {patterns[pattern]} and {idx}",
                            )
                        )
                    else:
                        patterns[pattern] = idx

                if fix:
                    changed, removed = _dedupe_identical_lines(ctx)
                    if changed:
                        fixes.append(Issue(repo, "sepolicy", "info", ctx, f"Removed {removed} exact duplicate lines"))

    return issues, fixes


def mine_audit_logs(root: Path) -> list[Issue]:
    issues: list[Issue] = []

    # Find audit-* directories.
    audit_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("audit-")]
    if not audit_dirs:
        return issues

    avc_re = re.compile(r"avc: denied\s+\{([^}]+)\}.*scontext=([^\s]+)\s+tcontext=([^\s]+)\s+tclass=([^\s]+)")
    unique: set[tuple[str, str, str, str]] = set()

    for d in audit_dirs:
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = _read_text(f)
            except Exception:
                continue
            for line in text.splitlines():
                m = avc_re.search(line)
                if not m:
                    continue
                perms = " ".join(m.group(1).split())
                sctx = m.group(2)
                tctx = m.group(3)
                tclass = m.group(4)
                unique.add((perms, sctx, tctx, tclass))

    if unique:
        # Summarize as info issues (report-only).
        for perms, sctx, tctx, tclass in sorted(unique):
            issues.append(
                Issue(
                    repo="(runtime)",
                    category="audit",
                    severity="info",
                    file=None,
                    message=f"AVC denied {{{perms}}} {tclass} scontext={sctx} tcontext={tctx}",
                )
            )

    return issues


def _group_by_repo(issues: list[Issue]) -> dict[str, list[Issue]]:
    out: dict[str, list[Issue]] = {}
    for i in issues:
        out.setdefault(i.repo, []).append(i)
    return out


def _print_report(issues: list[Issue], fixes: list[Issue]) -> None:
    by_repo = _group_by_repo(issues)

    def _fmt(i: Issue) -> str:
        loc = ""
        if i.file is not None:
            loc = f" ({i.file})"
        return f"[{i.severity.upper()}] {i.category}: {i.message}{loc}"

    print("=== Pre-build validation report ===")
    print(f"Issues: {len(issues)}  Fixes: {len(fixes)}")

    if fixes:
        print("\n-- Fixes applied (safe) --")
        for f in fixes:
            print(_fmt(f))

    for repo in sorted(by_repo):
        print(f"\n-- {repo} --")
        for i in sorted(by_repo[repo], key=lambda x: (x.severity, x.category, str(x.file or ""))):
            print(_fmt(i))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Tree root (contains android_device_*/android_vendor_*)")
    ap.add_argument("--repos", nargs="*", default=REPOS_DEFAULT)
    ap.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe autofixes (dedupe identical sepolicy context lines; remove exact-duplicate init service stanzas)",
    )
    ap.add_argument("--elf", action="store_true", help="Validate ELF DT_NEEDED closure for init-referenced /vendor binaries")
    ap.add_argument("--elf-max", type=int, default=200, help="Max number of unique binaries to readelf when --elf is enabled")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    repos: list[str] = list(args.repos)

    issues: list[Issue] = []
    fixes: list[Issue] = []

    index = build_index(root, repos)

    issues.extend(check_init_imports_and_services(root, repos, index))
    dup_issues, dup_fixes = check_init_duplicate_services(root, repos, fix=args.fix)
    issues.extend(dup_issues)
    fixes.extend(dup_fixes)
    issues.extend(check_proprietary_files(root, repos, index))
    issues.extend(check_json_files(root, repos))
    issues.extend(check_makefile_includes(root, repos))

    if args.elf:
        issues.extend(check_elf_dependency_closure(root, repos, max_binaries=max(0, args.elf_max)))

    se_issues, se_fixes = check_sepolicy_contexts(root, repos, fix=args.fix)
    issues.extend(se_issues)
    fixes.extend(se_fixes)

    issues.extend(mine_audit_logs(root))

    _print_report(issues, fixes)

    # Exit semantics: treat any errors as failure; warn/info still exit 1 because user asked for strict audit.
    has_issues = bool(issues)
    did_fix = bool(fixes)
    if has_issues or did_fix:
        return 1
    return 0


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        # Allow piping to tools like head without noisy tracebacks.
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover
        print(f"Internal error: {e}", file=sys.stderr)
        raise SystemExit(2)
