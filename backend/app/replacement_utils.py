import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Common candidate column names
COMPONENT_CANDIDATES = ["component", "part_number", "part_no", "pn", "component_id", "partnumber"]
ASSY_CANDIDATES = ["assy_pn", "assembly_id", "assembly", "assy", "assembly_pn"]
COST_CANDIDATES = ["cost", "unit_cost", "unitprice", "unit_price", "price", "rate"]
QTY_CANDIDATES = ["quantity", "qty", "qnty", "quantity_required", "required_qty"]
EAU_CANDIDATES = ["EAU", "eau", "annual_usage", "annual_quantity", "annual_qty"]

def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def build_component_lookup_from_weldment(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (assy_col, comp_col, cost_col) if detected (or None)."""
    cols = df.columns.tolist()
    assy_col = detect_column(cols, ASSY_CANDIDATES)
    comp_col = detect_column(cols, COMPONENT_CANDIDATES)
    cost_col = detect_column(cols, COST_CANDIDATES)
    return assy_col, comp_col, cost_col

def build_component_replacement_map(
    weld_df: pd.DataFrame,
    pairwise_records: List[dict],
    assy_col: Optional[str] = None,
    comp_col: Optional[str] = None,
    cost_col: Optional[str] = None
) -> Dict[str, str]:
    """
    For pairwise records (only 100% matches expected) build a mapping from old_part -> chosen_min_cost_part.
    Return mapping: {old_part: new_part}
    """
    if assy_col is None or comp_col is None or cost_col is None:
        assy_col, comp_col, cost_col = build_component_lookup_from_weldment(weld_df)
    if assy_col is None or comp_col is None:
        # cannot do component-level mapping
        return {}

    # Normalize cost column if missing
    if cost_col is None:
        weld_df["_inferred_cost"] = 0.0
        cost_col = "_inferred_cost"
    else:
        # coerce to numeric
        weld_df[cost_col] = pd.to_numeric(weld_df[cost_col], errors='coerce').fillna(0.0)

    # Create mapping container
    replacement_map = {}

    # For each pair that is 100% matched, try to align component rows by component identifier
    for rec in pairwise_records:
        try:
            match_pct = float(rec.get("match_percentage") or 0.0)
        except Exception:
            match_pct = 0.0
        if round(match_pct, 6) < 100.0:  # only exact 100%
            continue

        a = rec.get("bom_a")
        b = rec.get("bom_b")
        if not a or not b:
            continue

        # get rows for both assemblies
        rows_a = weld_df[weld_df[assy_col].astype(str) == str(a)]
        rows_b = weld_df[weld_df[assy_col].astype(str) == str(b)]

        if rows_a.empty or rows_b.empty:
            continue

        # If both assemblies list the same component IDs (identical sets), process component-level min cost
        # We'll take the union of component ids found under both assemblies and for every unique component id
        # choose the part with minimum cost among any rows with that component id across assemblies.
        # NOTE: if components differ by functional name, mapping may fail; we take simple approach here.
        comp_vals = set(rows_a[comp_col].astype(str).str.strip().dropna().unique().tolist() +
                        rows_b[comp_col].astype(str).str.strip().dropna().unique().tolist())

        # If comp_vals seems empty, skip
        if not comp_vals:
            continue

        for comp in comp_vals:
            # all rows across both assemblies having this comp
            candidate_rows = weld_df[weld_df[comp_col].astype(str).str.strip() == str(comp)]
            if candidate_rows.empty:
                continue
            # find row with min cost
            idx_min = candidate_rows[cost_col].astype(float).idxmin()
            min_part = candidate_rows.loc[idx_min, comp_col]
            # For all other part occurrences, map them to min_part
            other_parts = candidate_rows[comp_col].astype(str).unique().tolist()
            for op in other_parts:
                op = str(op)
                minp = str(min_part)
                if op == minp:
                    continue
                replacement_map[op] = minp

    return replacement_map

def apply_replacements_to_bomworkbook(
    bom_workbook: Dict[str, pd.DataFrame],
    replacement_map: Dict[str, str],
    qty_col_hint: Optional[str] = None,
    eau_value: Optional[float] = 1.0,
    cost_lookup: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """
    Apply replacements across all DataFrames (sheets) in bom_workbook (dict: sheetname -> df).
    Returns modified workbook dict and a list of replacements with savings summaries.
    cost_lookup: optional dict mapping part -> unit_cost (used if BOM sheets don't contain cost)
    """
    modified = {}
    replacements_summary = []

    for sheet_name, df in bom_workbook.items():
        if df is None or df.empty:
            modified[sheet_name] = df
            continue

        # detect quantity and component columns
        comp_col = detect_column(df.columns.tolist(), COMPONENT_CANDIDATES)
        qty_col = qty_col_hint or detect_column(df.columns.tolist(), QTY_CANDIDATES)
        cost_col = detect_column(df.columns.tolist(), COST_CANDIDATES)

        # If no component column present, skip sheet
        if comp_col is None:
            modified[sheet_name] = df
            continue

        df_copy = df.copy()

        # Add cost column if missing and cost_lookup provided
        if cost_col is None and cost_lookup:
            df_copy["_inferred_cost"] = df_copy[comp_col].map(lambda x: cost_lookup.get(str(x), 0.0))
            cost_col_use = "_inferred_cost"
        elif cost_col is not None:
            df_copy[cost_col] = pd.to_numeric(df_copy[cost_col], errors='coerce').fillna(0.0)
            cost_col_use = cost_col
        else:
            # no cost info available; treat cost as 0
            df_copy["_inferred_cost"] = 0.0
            cost_col_use = "_inferred_cost"

        # Apply replacements row-wise and compute savings
        changes = []
        for idx, row in df_copy.iterrows():
            comp_val = str(row[comp_col]).strip()
            if comp_val in replacement_map:
                new_comp = replacement_map[comp_val]
                old_cost = float(row.get(cost_col_use, 0.0) or 0.0)
                new_cost = float(cost_lookup.get(new_comp, 0.0) if cost_lookup else (df_copy[df_copy[comp_col].astype(str)==new_comp][cost_col_use].astype(float).head(1).squeeze() if new_comp in df_copy[comp_col].astype(str).values else 0.0))
                quantity = 1
                if qty_col and qty_col in df_copy.columns:
                    try:
                        quantity = float(row.get(qty_col, 1) or 1.0)
                    except Exception:
                        quantity = 1.0

                # EAU applied per-item per-year (EAU provided per endpoint/request)
                savings = (old_cost - new_cost) * (eau_value or 1.0) * quantity
                pct_saving = ((old_cost - new_cost) / old_cost * 100.0) if old_cost and old_cost > 0 else 0.0

                # Apply replacement
                df_copy.at[idx, comp_col] = new_comp

                changes.append({
                    "sheet": sheet_name,
                    "row_index": int(idx),
                    "old_part": comp_val,
                    "new_part": new_comp,
                    "old_unit_cost": old_cost,
                    "new_unit_cost": new_cost,
                    "quantity": quantity,
                    "EAU": eau_value,
                    "savings": float(savings),
                    "pct_savings": float(pct_saving)
                })

        modified[sheet_name] = df_copy

        # aggregate sheet-level changes into replacements_summary
        if changes:
            replacements_summary.extend(changes)

    # return modified workbook and summary list
    return modified, replacements_summary
