#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import pandas as pd
import requests

SYSTEM_PROMPT = """
You are an information extraction system for CeO2 synthesis papers.
Extract sample-level structured data from a single paper.

Rules:
1. Extract ONLY CeO2 samples reported in THIS paper, not background literature examples.
2. Split different samples into separate sample objects whenever the paper distinguishes them by:
   2.1 sample names (e.g. CeO2-R, CeO2-C, CeO2-O, sample A/B/C)
   2.2 different synthesis temperatures/times/additives
   2.3 different morphologies/facets
3. Do NOT guess missing values.
4. If explicit, use status='reported_directly'.
5. If strongly implied by immediate local context, use status='inferred_from_context'.
6. If absent or unclear, use status='not_reported'.
7. Morphology.primary must be one of:
   rod, cube, octahedron, truncated_octahedron, polyhedron, sphere, particle,
   wire, tube, irregular_particle, spindle, sheet, flower, hollow_sphere,
   mesoporous, nanobelt, unknown
   Put additional shapes in morphology.secondary.
8. Facets:
   8.1 Use Miller-index braces only: {111}, {110}, {100}, {001}, {210}, etc.
   8.2 Convert (111) style text to {111}.
   8.3 Put the dominant facet in facets.primary; put others in facets.secondary.
   8.4 Also fill facets.exposed with the full unique list of exposed facets.
9. Numeric fields (temperature_C, time_h, calcination_*, size_nm):
   9.1 Single value: set "value" to a number, leave "min" and "max" as null.
   9.2 Range (e.g. 100-200 C, 12-24 h): set "min" and "max", leave "value" as null.
   9.3 Do not invent numbers.
10. Preserve short evidence snippets, plain text only.
11. Return ONLY valid JSON.
""".strip()

USER_TEMPLATE = """
paper_id: {paper_id}
paper_title: {paper_title}
chunk_id: {chunk_id}

Target JSON schema:
{{
  "paper_id": "{paper_id}",
  "paper_title": "{paper_title}",
  "samples": [
    {{
      "sample_id": "",
      "sample_name_in_paper": "",
      "material": "CeO2",
      "synthesis_method": "",
      "precursor": {{"value": "", "status": "reported_directly | inferred_from_context | not_reported"}},
      "base_or_mineralizer": {{"value": "", "status": "reported_directly | inferred_from_context | not_reported"}},
      "additives": [
        {{"value": "", "status": "reported_directly | inferred_from_context | not_reported"}}
      ],
      "temperature_C": {{
        "value": null,
        "min": null,
        "max": null,
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "time_h": {{
        "value": null,
        "min": null,
        "max": null,
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "calcination_temperature_C": {{
        "value": null,
        "min": null,
        "max": null,
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "calcination_time_h": {{
        "value": null,
        "min": null,
        "max": null,
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "morphology": {{
        "primary": "rod | cube | octahedron | truncated_octahedron | polyhedron | sphere | particle | wire | tube | irregular_particle | spindle | sheet | flower | hollow_sphere | mesoporous | nanobelt | unknown",
        "secondary": [],
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "facets": {{
        "primary": "",
        "secondary": [],
        "exposed": [],
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "size_nm": {{
        "value": null,
        "min": null,
        "max": null,
        "status": "reported_directly | inferred_from_context | not_reported"
      }},
      "evidence": {{
        "synthesis_text": [],
        "morphology_text": [],
        "facet_text": []
      }},
      "notes": ""
    }}
  ]
}}

Input text:
{text}
""".strip()

MERGE_TEMPLATE = """
You are merging partial sample extraction results from multiple chunks of the SAME CeO2 paper.

Task:
1. Merge duplicate samples across chunks.
2. Prefer directly reported values over inferred values.
3. Keep separate samples separate unless they are clearly the same sample.
4. Keep missing values as missing.
5. Preserve numeric ranges (min/max), facet.exposed lists, and special morphologies.
6. Return ONLY valid JSON with the same target schema.

Paper id: {paper_id}
Paper title: {paper_title}

Partial extracted JSON objects:
{partial_jsons}
""".strip()

JSON_REPAIR_TEMPLATE = """
The previous output was intended to be JSON but is invalid.

Please convert the following content into STRICT VALID JSON only.
Do not add commentary. Do not add markdown fences.
Keep the same information if possible, but fix syntax so it parses as JSON.

Invalid content:
{bad_output}
""".strip()

MORPH_SET = {
    "rod",
    "cube",
    "octahedron",
    "truncated_octahedron",
    "polyhedron",
    "sphere",
    "particle",
    "wire",
    "tube",
    "irregular_particle",
    "spindle",
    "sheet",
    "flower",
    "hollow_sphere",
    "mesoporous",
    "nanobelt",
    "unknown",
}
COMMON_FACETS = {"{111}", "{110}", "{100}", "{001}", "{210}", "{311}", "{220}", "{200}"}
STATUS_SET = {"reported_directly", "inferred_from_context", "not_reported"}
NUMERIC_FIELD_NAMES = (
    "temperature_C",
    "time_h",
    "calcination_temperature_C",
    "calcination_time_h",
    "size_nm",
)

SECTION_HINTS = [
    "abstract",
    "introduction",
    "experimental",
    "materials and methods",
    "synthesis",
    "preparation",
    "results",
    "discussion",
    "results and discussion",
    "characterization",
    "conclusion",
]

KEYWORD_HINTS = [
    "CeO2",
    "ceria",
    "hydrothermal",
    "solvothermal",
    "calcined",
    "NaOH",
    "urea",
    "Na3PO4",
    "PVP",
    "rod",
    "cube",
    "octahedron",
    "polyhedron",
    "wire",
    "sphere",
    "particle",
    "spindle",
    "sheet",
    "flower",
    "{111}",
    "{110}",
    "{100}",
    "{001}",
    "(111)",
    "(110)",
    "(100)",
    "(001)",
]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_control_chars(s: str) -> str:
    return "".join(ch for ch in str(s) if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 0) -> str:
    """Extract PDF text. max_pages <= 0 means all pages."""
    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)
        limit = n_pages if max_pages <= 0 else min(max_pages, n_pages)
        texts = []
        for i in range(limit):
            try:
                page_text = doc.load_page(i).get_text("text")
            except Exception as exc:
                print(
                    f"WARNING: failed to read page {i + 1}/{n_pages} of {pdf_path.name}: {exc}",
                    flush=True,
                )
                continue
            if page_text:
                texts.append(page_text)
        if not texts:
            raise ValueError(f"No extractable text from {pdf_path.name}")
        return "\n".join(texts)
    finally:
        doc.close()


def guess_title_from_text(text: str, fallback: str) -> str:
    lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]
    bad = ("abstract", "keywords", "article history", "received", "accepted", "available online")
    for line in lines[:40]:
        low = line.lower()
        if any(low.startswith(b) for b in bad):
            continue
        if len(line) < 15 or len(line) > 180:
            continue
        if any(w in low for w in ["university", "department", "institute", "laboratory"]):
            continue
        return line
    return fallback


def select_relevant_text(text: str, max_chars: int = 0) -> str:
    """Select synthesis-related paragraphs. max_chars <= 0 means no truncation."""
    paragraphs = [normalize_space(p) for p in re.split(r"\n\s*\n", text) if normalize_space(p)]
    kept = []
    for p in paragraphs:
        low = p.lower()
        if any(h in low[:120] for h in SECTION_HINTS) or any(k.lower() in low for k in KEYWORD_HINTS):
            kept.append(p)
    if not kept:
        lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]
        chunk = []
        for line in lines:
            low = line.lower()
            if any(k.lower() in low for k in KEYWORD_HINTS) or any(h in low for h in SECTION_HINTS):
                chunk.append(line)
        kept = chunk if chunk else lines
    merged = "\n\n".join(kept)
    if max_chars > 0:
        return merged[:max_chars]
    return merged


def chunk_text(text: str, max_chunk_chars: int = 7000, max_chunks: int = 0) -> List[str]:
    """Split text into chunks. max_chunks <= 0 means keep all chunks."""
    paras = [normalize_space(p) for p in re.split(r"\n\s*\n", text) if normalize_space(p)]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for p in paras:
        if current_len + len(p) + 2 > max_chunk_chars and current:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        chunks.append("\n\n".join(current))
    if not chunks:
        if max_chunk_chars > 0:
            chunks = [text[i : i + max_chunk_chars] for i in range(0, max(len(text), 1), max_chunk_chars)]
        else:
            chunks = [text]
    if max_chunks > 0:
        return chunks[:max_chunks]
    return chunks


def request_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def strip_markdown_and_extract_json_block(s: str) -> str:
    s = clean_control_chars(s.strip())
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    return s.strip()


def try_parse_json(s: str) -> Tuple[bool, Any, str]:
    cleaned = strip_markdown_and_extract_json_block(s)
    try:
        return True, json.loads(cleaned), cleaned
    except Exception:
        pass
    repaired = cleaned.replace("\r", " ").replace("\x00", " ")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = repaired.replace("\t", " ")
    repaired = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", repaired)
    try:
        return True, json.loads(repaired), repaired
    except Exception:
        return False, None, repaired


def validate_extraction_obj_loose(obj: Any) -> Dict[str, Any]:
    """Minimal checks right after model JSON parse (before normalization)."""
    if not isinstance(obj, dict):
        raise TypeError(f"root must be object, got {type(obj).__name__}")
    if "samples" not in obj:
        raise TypeError("root missing 'samples'")
    if not isinstance(obj["samples"], list):
        raise TypeError(f"samples must be list, got {type(obj['samples']).__name__}")
    for i, sample in enumerate(obj["samples"]):
        if not isinstance(sample, dict):
            raise TypeError(f"samples[{i}]: expected object, got {type(sample).__name__}")
    return obj


def parse_or_repair(
    api_base: str, api_key: str, model: str, text_output: str, max_tokens: int = 4000
) -> Dict[str, Any]:
    ok, obj, cleaned = try_parse_json(text_output)
    if ok:
        return validate_extraction_obj_loose(obj)
    second = request_chat(
        api_base,
        api_key,
        model,
        [
            {"role": "system", "content": "You repair malformed JSON. Return only strict valid JSON."},
            {"role": "user", "content": JSON_REPAIR_TEMPLATE.format(bad_output=cleaned)},
        ],
        0.0,
        max_tokens,
    )
    ok2, obj2, cleaned2 = try_parse_json(second)
    if ok2:
        return validate_extraction_obj_loose(obj2)
    raise ValueError(f"JSON parse failed after repair. First 500 chars: {cleaned2[:500]}")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _validate_status(status: Any, path: str) -> None:
    if not isinstance(status, str):
        raise TypeError(f"{path}: status must be str, got {type(status).__name__}")
    if status not in STATUS_SET:
        raise ValueError(f"{path}: invalid status {status!r}")


def _validate_reported_value(field: Any, path: str, *, allow_empty_str: bool = True) -> None:
    if not isinstance(field, dict):
        raise TypeError(f"{path}: expected object, got {type(field).__name__}")
    if "value" not in field or "status" not in field:
        raise TypeError(f"{path}: must contain 'value' and 'status'")
    _validate_status(field["status"], f"{path}.status")
    value = field["value"]
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{path}.value: expected str or null, got {type(value).__name__}")
    if not allow_empty_str and not value.strip():
        raise ValueError(f"{path}.value: empty string not allowed when present")


def _validate_numeric_field(field: Any, path: str) -> None:
    if not isinstance(field, dict):
        raise TypeError(f"{path}: expected object, got {type(field).__name__}")
    for key in ("value", "min", "max", "status"):
        if key not in field:
            raise TypeError(f"{path}: missing key {key!r}")
    _validate_status(field["status"], f"{path}.status")
    for key in ("value", "min", "max"):
        v = field[key]
        if v is None:
            continue
        if not _is_number(v):
            raise TypeError(f"{path}.{key}: expected number or null, got {type(v).__name__}")
    vmin, vmax, val = field["min"], field["max"], field["value"]
    if vmin is not None and vmax is not None and float(vmin) > float(vmax):
        raise ValueError(f"{path}: min ({vmin}) > max ({vmax})")
    if val is not None and (vmin is not None or vmax is not None):
        # allow value together with range only if value lies inside [min, max]
        lo = float(vmin) if vmin is not None else float("-inf")
        hi = float(vmax) if vmax is not None else float("inf")
        if not (lo <= float(val) <= hi):
            raise ValueError(f"{path}: value {val} outside min/max range")


def validate_extraction_obj(obj: Any) -> Dict[str, Any]:
    """Strict structural/type checks on model JSON."""
    if not isinstance(obj, dict):
        raise TypeError(f"root must be object, got {type(obj).__name__}")
    if "samples" not in obj:
        raise TypeError("root missing 'samples'")
    samples = obj["samples"]
    if not isinstance(samples, list):
        raise TypeError(f"samples must be list, got {type(samples).__name__}")

    for i, sample in enumerate(samples):
        path = f"samples[{i}]"
        if not isinstance(sample, dict):
            raise TypeError(f"{path}: expected object, got {type(sample).__name__}")

        for key in (
            "sample_id",
            "sample_name_in_paper",
            "material",
            "synthesis_method",
            "precursor",
            "base_or_mineralizer",
            "additives",
            "temperature_C",
            "time_h",
            "calcination_temperature_C",
            "calcination_time_h",
            "morphology",
            "facets",
            "size_nm",
            "evidence",
            "notes",
        ):
            if key not in sample:
                raise TypeError(f"{path}: missing key {key!r}")

        for key in ("sample_id", "sample_name_in_paper", "material", "synthesis_method", "notes"):
            if not isinstance(sample[key], str):
                raise TypeError(f"{path}.{key}: expected str, got {type(sample[key]).__name__}")

        _validate_reported_value(sample["precursor"], f"{path}.precursor")
        _validate_reported_value(sample["base_or_mineralizer"], f"{path}.base_or_mineralizer")

        additives = sample["additives"]
        if not isinstance(additives, list):
            raise TypeError(f"{path}.additives: expected list")
        for j, add in enumerate(additives):
            _validate_reported_value(add, f"{path}.additives[{j}]")

        for num_key in NUMERIC_FIELD_NAMES:
            _validate_numeric_field(sample[num_key], f"{path}.{num_key}")

        morph = sample["morphology"]
        if not isinstance(morph, dict):
            raise TypeError(f"{path}.morphology: expected object")
        if not isinstance(morph.get("primary"), str):
            raise TypeError(f"{path}.morphology.primary: expected str")
        if morph["primary"] not in MORPH_SET:
            raise ValueError(f"{path}.morphology.primary: invalid {morph['primary']!r}")
        if not isinstance(morph.get("secondary"), list):
            raise TypeError(f"{path}.morphology.secondary: expected list")
        for j, m in enumerate(morph["secondary"]):
            if not isinstance(m, str) or m not in MORPH_SET:
                raise ValueError(f"{path}.morphology.secondary[{j}]: invalid {m!r}")
        _validate_status(morph.get("status"), f"{path}.morphology.status")

        facets = sample["facets"]
        if not isinstance(facets, dict):
            raise TypeError(f"{path}.facets: expected object")
        for key in ("primary", "secondary", "exposed", "status"):
            if key not in facets:
                raise TypeError(f"{path}.facets: missing key {key!r}")
        if not isinstance(facets["primary"], str):
            raise TypeError(f"{path}.facets.primary: expected str")
        if facets["primary"] and not re.fullmatch(r"\{\d{3}\}", facets["primary"]):
            raise ValueError(f"{path}.facets.primary: invalid facet {facets['primary']!r}")
        for key in ("secondary", "exposed"):
            if not isinstance(facets[key], list):
                raise TypeError(f"{path}.facets.{key}: expected list")
            for j, f in enumerate(facets[key]):
                if not isinstance(f, str) or not re.fullmatch(r"\{\d{3}\}", f):
                    raise ValueError(f"{path}.facets.{key}[{j}]: invalid facet {f!r}")
        _validate_status(facets["status"], f"{path}.facets.status")

        evidence = sample["evidence"]
        if not isinstance(evidence, dict):
            raise TypeError(f"{path}.evidence: expected object")
        for key in ("synthesis_text", "morphology_text", "facet_text"):
            if key not in evidence or not isinstance(evidence[key], list):
                raise TypeError(f"{path}.evidence.{key}: expected list")
            for j, snip in enumerate(evidence[key]):
                if not isinstance(snip, str):
                    raise TypeError(f"{path}.evidence.{key}[{j}]: expected str")

    return obj


def call_llm_chunk(
    api_base: str,
    api_key: str,
    model: str,
    paper_id: str,
    paper_title: str,
    chunk_id: str,
    text: str,
    temperature: float = 0.0,
    max_tokens: int = 3000,
) -> Dict[str, Any]:
    first = request_chat(
        api_base,
        api_key,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    chunk_id=chunk_id,
                    text=text,
                ),
            },
        ],
        temperature,
        max_tokens,
    )
    return parse_or_repair(api_base, api_key, model, first, max_tokens=max_tokens)


def _merge_once(
    api_base: str,
    api_key: str,
    model: str,
    paper_id: str,
    paper_title: str,
    partial_objs: List[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    partial_text = json.dumps(partial_objs, ensure_ascii=False, indent=2)
    out = request_chat(
        api_base,
        api_key,
        model,
        [
            {
                "role": "system",
                "content": "You merge partial sample extraction results into one final strict JSON object.",
            },
            {
                "role": "user",
                "content": MERGE_TEMPLATE.format(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    partial_jsons=partial_text,
                ),
            },
        ],
        0.0,
        max_tokens,
    )
    return parse_or_repair(api_base, api_key, model, out, max_tokens=max_tokens)


def merge_partial_results(
    api_base: str,
    api_key: str,
    model: str,
    paper_id: str,
    paper_title: str,
    partial_objs: List[Dict[str, Any]],
    max_tokens: int = 4000,
    merge_batch_size: int = 4,
) -> Dict[str, Any]:
    """Merge all partial JSONs without character truncation; batch recursively if many."""
    if not partial_objs:
        raise ValueError("No partial objects to merge.")
    if len(partial_objs) == 1:
        return validate_extraction_obj_loose(partial_objs[0])

    current = list(partial_objs)
    while len(current) > 1:
        if len(current) <= merge_batch_size:
            return _merge_once(
                api_base, api_key, model, paper_id, paper_title, current, max_tokens
            )
        next_level: List[Dict[str, Any]] = []
        for i in range(0, len(current), merge_batch_size):
            batch = current[i : i + merge_batch_size]
            if len(batch) == 1:
                next_level.append(batch[0])
            else:
                next_level.append(
                    _merge_once(
                        api_base, api_key, model, paper_id, paper_title, batch, max_tokens
                    )
                )
        current = next_level
    return validate_extraction_obj_loose(current[0])


def normalize_status(x: Any) -> str:
    s = str(x).strip() if x is not None else "not_reported"
    return s if s in STATUS_SET else "not_reported"


def normalize_chemical_text(x: Any) -> str:
    if not x:
        return ""
    x = str(x).strip()
    x = x.replace("$", "·").replace("•", "·").replace("−", "-").replace("–", "-")
    x = re.sub(r"\s+", " ", x)
    x = x.replace("Ce(NO3)3 6H2O", "Ce(NO3)3·6H2O")
    x = x.replace("CeCl3 7H2O", "CeCl3·7H2O")
    return x


def normalize_morph(x: Any) -> str:
    if not x:
        return "unknown"
    x = str(x).strip()
    if x in MORPH_SET:
        return x
    low = x.lower().replace("_", " ").replace("-", " ")
    low = re.sub(r"\s+", " ", low).strip()
    mapping = {
        "nanorod": "rod",
        "nanorods": "rod",
        "rod like": "rod",
        "rodlike": "rod",
        "nanocube": "cube",
        "nanocubes": "cube",
        "cubic": "cube",
        "octahedral": "octahedron",
        "nano octahedrons": "octahedron",
        "nano octahedron": "octahedron",
        "truncated octahedron": "truncated_octahedron",
        "truncated octahedra": "truncated_octahedron",
        "nanopolyhedra": "polyhedron",
        "polyhedral": "polyhedron",
        "spherical": "sphere",
        "nanosphere": "sphere",
        "nanospheres": "sphere",
        "hollow sphere": "hollow_sphere",
        "hollow spheres": "hollow_sphere",
        "nanoparticle": "particle",
        "nanoparticles": "particle",
        "particle like": "particle",
        "nanowire": "wire",
        "nanowires": "wire",
        "nanotube": "tube",
        "nanotubes": "tube",
        "irregular particle": "irregular_particle",
        "irregular particles": "irregular_particle",
        "nanosheet": "sheet",
        "nanosheets": "sheet",
        "sheet": "sheet",
        "sheets": "sheet",
        "spindle": "spindle",
        "spindle like": "spindle",
        "flower": "flower",
        "flower like": "flower",
        "nanoflower": "flower",
        "mesoporous": "mesoporous",
        "nanobelt": "nanobelt",
        "nanobelts": "nanobelt",
    }
    return mapping.get(low, "unknown")


def normalize_facet(x: Any) -> str:
    if not x:
        return ""
    s = str(x).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\((\d{3})\)", r"{\1}", s)
    s = re.sub(r"(?<!\{)(\d{3})(?!\})", r"{\1}", s)
    s = s.replace(" facet", "").replace("facets", "").strip()
    m = re.search(r"\{\d{3}\}", s)
    if not m:
        return ""
    return m.group(0)


def normalize_facet_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = re.split(r"[;,/|]\s*", values)
        values = parts
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for v in values:
        f = normalize_facet(v)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _to_optional_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    if isinstance(x, str):
        s = x.strip().replace("°C", "").replace("C", "").replace("h", "").strip()
        # range written as "100-200" or "100–200"
        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*[-–~to]+\s*([+-]?\d+(?:\.\d+)?)", s, flags=re.I)
        if m:
            return None  # handled by range parser
        try:
            return float(s)
        except ValueError:
            return None
    if _is_number(x):
        return float(x)
    return None


def normalize_numeric_field(field: Any) -> Dict[str, Any]:
    """Normalize single value or min/max range."""
    if field is None or field == "":
        return {"value": None, "min": None, "max": None, "status": "not_reported"}
    if _is_number(field) or (isinstance(field, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", field.strip())):
        return {
            "value": _to_optional_float(field),
            "min": None,
            "max": None,
            "status": "reported_directly",
        }
    if not isinstance(field, dict):
        return {"value": None, "min": None, "max": None, "status": "not_reported"}

    status = normalize_status(field.get("status", "not_reported"))
    value = field.get("value", None)
    vmin = field.get("min", field.get("value_min", None))
    vmax = field.get("max", field.get("value_max", None))

    # string range inside value
    if isinstance(value, str):
        m = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)\s*[-–~to]+\s*([+-]?\d+(?:\.\d+)?)",
            value.strip(),
            flags=re.I,
        )
        if m:
            vmin = float(m.group(1))
            vmax = float(m.group(2))
            value = None

    value_f = _to_optional_float(value)
    min_f = _to_optional_float(vmin)
    max_f = _to_optional_float(vmax)
    if min_f is not None and max_f is not None and min_f > max_f:
        min_f, max_f = max_f, min_f
    if value_f is not None and min_f is None and max_f is None:
        return {"value": value_f, "min": None, "max": None, "status": status}
    if min_f is not None or max_f is not None:
        return {"value": None, "min": min_f, "max": max_f, "status": status}
    return {"value": value_f, "min": None, "max": None, "status": status}


def repair_record(obj: Dict[str, Any], paper_id: str, paper_title: str) -> Dict[str, Any]:
    out = {"paper_id": paper_id, "paper_title": paper_title, "samples": []}
    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    seen = set()
    for idx, s in enumerate(samples, start=1):
        if not isinstance(s, dict):
            raise TypeError(f"samples[{idx - 1}] is not an object")

        precursor = s.get("precursor", {}) or {}
        base = s.get("base_or_mineralizer", {}) or {}
        morph = s.get("morphology", {}) or {}
        facets = s.get("facets", {}) or {}
        evidence = s.get("evidence", {}) or {}
        additives = s.get("additives", []) or []

        sample_id = s.get("sample_id") or f"{paper_id}__S{idx:02d}"
        if sample_id in seen:
            sample_id = f"{paper_id}__S{idx:02d}"
        seen.add(sample_id)

        primary_morph = normalize_morph(morph.get("primary", "unknown"))
        secondary_morph = []
        for x in morph.get("secondary", []) or []:
            m = normalize_morph(x)
            if m not in ("unknown", primary_morph) and m not in secondary_morph:
                secondary_morph.append(m)

        primary_facet = normalize_facet(facets.get("primary", ""))
        secondary_facets = normalize_facet_list(facets.get("secondary", []))
        exposed = normalize_facet_list(facets.get("exposed", []))
        if not exposed:
            exposed = []
            for f in [primary_facet, *secondary_facets]:
                if f and f not in exposed:
                    exposed.append(f)
        secondary_facets = [f for f in secondary_facets if f != primary_facet]
        if primary_facet and primary_facet not in exposed:
            exposed.insert(0, primary_facet)

        fixed = {
            "sample_id": sample_id,
            "sample_name_in_paper": str(s.get("sample_name_in_paper", "") or ""),
            "material": "CeO2",
            "synthesis_method": str(s.get("synthesis_method", "") or ""),
            "precursor": {
                "value": normalize_chemical_text(
                    precursor.get("value", "") if isinstance(precursor, dict) else precursor
                ),
                "status": normalize_status(
                    precursor.get("status", "not_reported") if isinstance(precursor, dict) else "not_reported"
                ),
            },
            "base_or_mineralizer": {
                "value": normalize_chemical_text(
                    base.get("value", "") if isinstance(base, dict) else base
                ),
                "status": normalize_status(
                    base.get("status", "not_reported") if isinstance(base, dict) else "not_reported"
                ),
            },
            "additives": [],
            "temperature_C": normalize_numeric_field(s.get("temperature_C")),
            "time_h": normalize_numeric_field(s.get("time_h")),
            "calcination_temperature_C": normalize_numeric_field(s.get("calcination_temperature_C")),
            "calcination_time_h": normalize_numeric_field(s.get("calcination_time_h")),
            "morphology": {
                "primary": primary_morph,
                "secondary": secondary_morph,
                "status": normalize_status(morph.get("status", "not_reported")),
            },
            "facets": {
                "primary": primary_facet,
                "secondary": secondary_facets,
                "exposed": exposed,
                "status": normalize_status(facets.get("status", "not_reported")),
            },
            "size_nm": normalize_numeric_field(s.get("size_nm")),
            "evidence": {
                "synthesis_text": [
                    clean_control_chars(str(x))[:300]
                    for x in (evidence.get("synthesis_text", []) or [])
                ],
                "morphology_text": [
                    clean_control_chars(str(x))[:300]
                    for x in (evidence.get("morphology_text", []) or [])
                ],
                "facet_text": [
                    clean_control_chars(str(x))[:300]
                    for x in (evidence.get("facet_text", []) or [])
                ],
            },
            "notes": clean_control_chars(s.get("notes", "") or ""),
        }
        for a in additives:
            if isinstance(a, dict):
                fixed["additives"].append(
                    {
                        "value": normalize_chemical_text(a.get("value", "") or ""),
                        "status": normalize_status(a.get("status", "not_reported")),
                    }
                )
            elif isinstance(a, str) and a.strip():
                fixed["additives"].append(
                    {"value": normalize_chemical_text(a), "status": "reported_directly"}
                )
        out["samples"].append(fixed)

    return validate_extraction_obj(out)


def _format_numeric_for_csv(field: Dict[str, Any]) -> Union[float, str, None]:
    if field.get("value") is not None:
        return field["value"]
    if field.get("min") is not None or field.get("max") is not None:
        lo = field.get("min")
        hi = field.get("max")
        if lo is not None and hi is not None:
            return f"{lo}-{hi}"
        if lo is not None:
            return f">={lo}"
        if hi is not None:
            return f"<={hi}"
    return None


def flatten_samples(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for s in obj.get("samples", []):
        rows.append(
            {
                "paper_id": obj.get("paper_id", ""),
                "paper_title": obj.get("paper_title", ""),
                "sample_id": s.get("sample_id", ""),
                "sample_name_in_paper": s.get("sample_name_in_paper", ""),
                "material": s.get("material", "CeO2"),
                "synthesis_method": s.get("synthesis_method", ""),
                "precursor_value": s["precursor"]["value"],
                "precursor_status": s["precursor"]["status"],
                "base_or_mineralizer_value": s["base_or_mineralizer"]["value"],
                "base_or_mineralizer_status": s["base_or_mineralizer"]["status"],
                "additives": "; ".join(
                    [a["value"] for a in s.get("additives", []) if a.get("value")]
                ),
                "temperature_C": _format_numeric_for_csv(s["temperature_C"]),
                "temperature_C_min": s["temperature_C"].get("min"),
                "temperature_C_max": s["temperature_C"].get("max"),
                "temperature_status": s["temperature_C"]["status"],
                "time_h": _format_numeric_for_csv(s["time_h"]),
                "time_h_min": s["time_h"].get("min"),
                "time_h_max": s["time_h"].get("max"),
                "time_status": s["time_h"]["status"],
                "calcination_temperature_C": _format_numeric_for_csv(
                    s["calcination_temperature_C"]
                ),
                "calcination_temperature_C_min": s["calcination_temperature_C"].get("min"),
                "calcination_temperature_C_max": s["calcination_temperature_C"].get("max"),
                "calcination_temperature_status": s["calcination_temperature_C"]["status"],
                "calcination_time_h": _format_numeric_for_csv(s["calcination_time_h"]),
                "calcination_time_h_min": s["calcination_time_h"].get("min"),
                "calcination_time_h_max": s["calcination_time_h"].get("max"),
                "calcination_time_status": s["calcination_time_h"]["status"],
                "morphology_primary": s["morphology"]["primary"],
                "morphology_secondary": "; ".join(s["morphology"]["secondary"]),
                "morphology_status": s["morphology"]["status"],
                "facets_primary": s["facets"]["primary"],
                "facets_secondary": "; ".join(s["facets"]["secondary"]),
                "facets_exposed": "; ".join(s["facets"]["exposed"]),
                "facets_status": s["facets"]["status"],
                "size_nm": _format_numeric_for_csv(s["size_nm"]),
                "size_nm_min": s["size_nm"].get("min"),
                "size_nm_max": s["size_nm"].get("max"),
                "size_status": s["size_nm"]["status"],
                "evidence_synthesis_text": " || ".join(s["evidence"]["synthesis_text"]),
                "evidence_morphology_text": " || ".join(s["evidence"]["morphology_text"]),
                "evidence_facet_text": " || ".join(s["evidence"]["facet_text"]),
                "notes": s.get("notes", ""),
            }
        )
    return rows


def process_one_pdf(path: Path, args):
    raw_text = extract_text_from_pdf(path, max_pages=args.max_pages)
    selected_text = select_relevant_text(raw_text, max_chars=args.max_chars)
    chunks = chunk_text(
        selected_text,
        max_chunk_chars=args.max_chunk_chars,
        max_chunks=args.max_chunks,
    )
    paper_id = path.stem
    paper_title = guess_title_from_text(raw_text, fallback=paper_id)

    partial_objs: List[Dict[str, Any]] = []
    chunk_errors: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = f"C{i:02d}"
        try:
            obj = call_llm_chunk(
                args.api_base,
                args.api_key,
                args.model,
                paper_id,
                paper_title,
                chunk_id,
                chunk,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            partial_objs.append(obj)
        except Exception as exc:
            err = f"{path.name} {chunk_id}: {exc}"
            print(f"ERROR: chunk extraction failed: {err}", flush=True)
            traceback.print_exc()
            chunk_errors.append(err)
            if args.fail_on_chunk_error:
                raise RuntimeError(err) from exc

    if not partial_objs:
        raise ValueError(
            "All chunk extraction attempts failed.\n" + "\n".join(chunk_errors)
        )
    if chunk_errors:
        print(
            f"WARNING: {len(chunk_errors)}/{len(chunks)} chunk(s) failed; "
            f"continuing with {len(partial_objs)} successful chunk(s).",
            flush=True,
        )

    if len(partial_objs) == 1:
        merged = partial_objs[0]
    else:
        merged = merge_partial_results(
            args.api_base,
            args.api_key,
            args.model,
            paper_id,
            paper_title,
            partial_objs,
            max_tokens=args.max_tokens,
            merge_batch_size=args.merge_batch_size,
        )

    fixed = repair_record(merged, paper_id, paper_title)
    return fixed, raw_text, selected_text, partial_objs, chunk_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--api_key", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument(
        "--max_pages",
        type=int,
        default=0,
        help="Max PDF pages to read; <=0 means all pages (default: all).",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Max chars after relevance filtering; <=0 means no truncation (default: no truncation).",
    )
    ap.add_argument("--max_chunk_chars", type=int, default=7000)
    ap.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Max chunks per paper; <=0 means keep all chunks (default: all).",
    )
    ap.add_argument(
        "--merge_batch_size",
        type=int,
        default=4,
        help="Batch size for recursive merge without truncating JSON text.",
    )
    ap.add_argument(
        "--fail_on_chunk_error",
        action="store_true",
        help="Stop the paper immediately when any chunk fails (default: record error and continue).",
    )
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    debug_dir = output_dir / "debug_text"
    partial_dir = output_dir / "partial_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    rows, logs = [], []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    total_pdfs = len(pdf_paths)
    for idx, path in enumerate(pdf_paths, start=1):
        print(f"Processing paper {idx}/{total_pdfs}: {path.name}", flush=True)
        try:
            fixed, raw_text, selected_text, partial_objs, chunk_errors = process_one_pdf(
                path, args
            )
            (json_dir / f"{path.stem}.json").write_text(
                json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (debug_dir / f"{path.stem}.raw.txt").write_text(raw_text, encoding="utf-8")
            (debug_dir / f"{path.stem}.selected.txt").write_text(
                selected_text, encoding="utf-8"
            )
            (partial_dir / f"{path.stem}.partial.json").write_text(
                json.dumps(partial_objs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if chunk_errors:
                (partial_dir / f"{path.stem}.chunk_errors.txt").write_text(
                    "\n".join(chunk_errors), encoding="utf-8"
                )
            rows.extend(flatten_samples(fixed))
            logs.append(
                {
                    "file": path.name,
                    "status": "ok_with_chunk_errors" if chunk_errors else "ok",
                    "samples": len(fixed.get("samples", [])),
                    "chunk_errors": len(chunk_errors),
                    "error": " | ".join(chunk_errors),
                }
            )
        except Exception as e:
            print(f"ERROR: paper failed: {path.name}: {e}", flush=True)
            traceback.print_exc()
            logs.append(
                {
                    "file": path.name,
                    "status": "failed",
                    "samples": 0,
                    "chunk_errors": 0,
                    "error": str(e),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "samples_extracted.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(logs).to_csv(output_dir / "run_log.csv", index=False, encoding="utf-8-sig")
    print(f"Done. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
