import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import math
import re
from collections import defaultdict

def clean_column_name(column_name: str) -> str:
    """Same cleaning helper used in BOM logic"""
    if pd.isna(column_name) or column_name is None:
        return "unknown"
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', str(column_name).lower())
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

# --- BOM preprocessing ---
def preprocess_bom_file(bom_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess BOM file to create assembly_id from Lev hierarchy"""
    bom_df.columns = bom_df.columns.str.lower().str.strip()

    # Create assembly IDs based on lev
    current_assembly = None
    assembly_ids = []
    for idx, row in bom_df.iterrows():
        lev = row.get('lev', None)
        component = row.get('component', None)
        if lev == 0:
            current_assembly = component
            assembly_ids.append(current_assembly)
        else:
            if current_assembly is None:
                current_assembly = f"ASSY_{idx}"
            assembly_ids.append(current_assembly)

    bom_df['assembly_id'] = assembly_ids
    bom_df['is_assembly'] = bom_df['lev'] == 0
    return bom_df


# --- BOM validators/parsers ---
def validate_bom_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean BOM data (keeps behavior from original main.py)"""
    print("Validating BOM data...")
    print("BOM columns:", df.columns.tolist())
    df.columns = [clean_column_name(col) for col in df.columns]

    # Find required columns
    component_col = None
    lev_col = None
    quantity_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'component' in col_lower or 'part' in col_lower:
            component_col = col
        elif 'lev' in col_lower or 'level' in col_lower:
            lev_col = col
        elif 'quantity' in col_lower or 'qty' in col_lower:
            quantity_col = col

    if component_col:
        df = df.rename(columns={component_col: 'component'})
    if lev_col:
        df = df.rename(columns={lev_col: 'lev'})
    if quantity_col:
        df = df.rename(columns={quantity_col: 'quantity'})

    missing_columns = []
    if 'component' not in df.columns:
        missing_columns.append('component')
    if 'lev' not in df.columns:
        missing_columns.append('lev')
    if 'quantity' not in df.columns:
        missing_columns.append('quantity')

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")

    df = df.dropna(subset=['component'])
    df['lev'] = pd.to_numeric(df['lev'], errors='coerce')
    df['quantity'] = pd.to_numeric(df.get('quantity', 0), errors='coerce').fillna(0)

    if 'assembly_id' not in df.columns:
        level_0_components = df[df['lev'] == 0]['component'].unique()
        if len(level_0_components) > 0:
            df['assembly_id'] = df['component'].apply(
                lambda x: next((comp for comp in level_0_components if str(comp) in str(x)), 'default_assembly')
            )
        else:
            df['assembly_id'] = 'default_assembly'

    print(f"BOM data validated. Records: {len(df)}")
    return df


# --- BOM analysis logic (quantity-aware) ---
def create_empty_bom_results() -> Dict[str, Any]:
    return {
        "similarity_matrix": {},
        "similar_pairs": [],
        "replacement_suggestions": [],
        "component_replacement_table": [],  # NEW: component-level replacement suggestions
        "bom_statistics": {
            "total_components": 0,
            "unique_components": 0,
            "total_assemblies": 0,
            "total_clusters": 0,
            "similar_pairs_count": 0,
            "reduction_potential": 0.0
        },
        "clusters": []
    }


def _to_float_safe(v) -> float:
    """Convert value to float safely; non-numeric -> 0.0"""
    try:
        if v is None:
            return 0.0
        # If it's already numeric
        if isinstance(v, (int, float)):
            return float(v)
        # If it's string, strip and parse
        s = str(v).strip()
        # empty string -> 0
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _compute_quantity_aware_lists(
    comp_map_a: Dict[str, float], comp_map_b: Dict[str, float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    Given two component->qty maps, compute:
      - common_components: list of {component, qty_a, qty_b, common_qty}
      - unique_components_a: list of {component, qty} where qty is remaining in A after common removed
      - unique_components_b: same for B
      - common_quantity_total: sum of common_qty across components

    All quantities are floats.
    """
    keys = sorted(set(list(comp_map_a.keys()) + list(comp_map_b.keys())))
    common_components = []
    unique_components_a = []
    unique_components_b = []
    common_quantity_total = 0.0

    for comp in keys:
        qA = _to_float_safe(comp_map_a.get(comp, 0.0))
        qB = _to_float_safe(comp_map_b.get(comp, 0.0))
        common_qty = min(qA, qB)

        if common_qty > 0:
            common_components.append({
                "component": comp,
                "qty_a": qA,
                "qty_b": qB,
                "common_qty": common_qty
            })
            common_quantity_total += common_qty

        remA = max(0.0, qA - common_qty)
        remB = max(0.0, qB - common_qty)

        if remA > 0:
            unique_components_a.append({
                "component": comp,
                "qty": remA
            })
        if remB > 0:
            unique_components_b.append({
                "component": comp,
                "qty": remB
            })

    return common_components, unique_components_a, unique_components_b, common_quantity_total


def compute_bom_similarity(assembly_components: Dict[str, Dict[str, Any]], threshold: float = 0.0) -> Dict[str, Any]:
    """
    Compute pairwise BOM similarity using STANDARD JACCARD on component NAMES (presence-only),
    and additionally prepare quantity-aware lists for UI display.

    Input:
      assembly_components: { assembly_id: { component_name: quantity, ... }, ... }
        - quantity may be numeric or string numeric; missing entries treated as 0
      threshold: minimum Jaccard percent (0..100) to include a pair in `similar_pairs`

    Output:
      {
        "similarity_matrix": { assy1: { assy2: jaccard_percent, ... }, ... },
        "similar_pairs": [
           {
             "bom_a": assy1,
             "bom_b": assy2,
             "similarity_score": jaccard_fraction,   # 0..1 (frontend expects fraction)
             "common_components": [{component, qty_a, qty_b, common_qty}, ...],
             "unique_components_a": [{component, qty}, ...],
             "unique_components_b": [{component, qty}, ...],
             "common_count": int,
             "common_quantity_total": float,
             "unique_count_a": int,
             "unique_count_b": int
           }, ...
        ]
      }
    """
    assemblies = list(assembly_components.keys())
    similarity_matrix: Dict[str, Dict[str, float]] = {}
    similar_pairs: List[Dict[str, Any]] = []

    # Defensive: if no assemblies, return empty structure
    if not assemblies:
        return {"similarity_matrix": {}, "similar_pairs": []}

    for i, assy_a in enumerate(assemblies):
        comp_map_a_raw = assembly_components.get(assy_a) or {}
        # Ensure comp_map is mapping component->float_qty
        comp_map_a = {
            str(k): _to_float_safe(v)
            for k, v in (comp_map_a_raw.items() if isinstance(comp_map_a_raw, dict) else [])
        }
        set_a = set(comp_map_a.keys())
        similarity_matrix.setdefault(assy_a, {})

        for j, assy_b in enumerate(assemblies):
            comp_map_b_raw = assembly_components.get(assy_b) or {}
            comp_map_b = {
                str(k): _to_float_safe(v)
                for k, v in (comp_map_b_raw.items() if isinstance(comp_map_b_raw, dict) else [])
            }
            set_b = set(comp_map_b.keys())

            # Standard Jaccard on presence-only sets (expressed as percent 0..100)
            if not set_a and not set_b:
                jaccard_pct = 100.0
            elif not set_a or not set_b:
                jaccard_pct = 0.0
            else:
                inter = set_a & set_b
                union = set_a | set_b
                jaccard_pct = (len(inter) / len(union)) * 100.0

            # store matrix value (rounded to reasonable precision)
            similarity_matrix[assy_a][assy_b] = round(jaccard_pct, 6)

            # store pair once (i < j) and only if meets threshold
            if i < j and jaccard_pct >= threshold:
                # compute quantity-aware lists for display only
                common_components, unique_components_a, unique_components_b, common_quantity_total = \
                    _compute_quantity_aware_lists(comp_map_a, comp_map_b)

                similar_pairs.append({
                    "bom_a": assy_a,
                    "bom_b": assy_b,
                    # frontend expects 0..1 fraction for progress bar
                    "similarity_score": round(jaccard_pct / 100.0, 6),
                    "common_components": common_components,
                    "unique_components_a": unique_components_a,
                    "unique_components_b": unique_components_b,
                    "common_count": len(common_components),
                    "common_quantity_total": common_quantity_total,
                    "unique_count_a": len(unique_components_a),
                    "unique_count_b": len(unique_components_b)
                })

    return {
        "similarity_matrix": similarity_matrix,
        "similar_pairs": similar_pairs
    }


def _compute_replacement_rows_for_pair(
    bom_a: str,
    bom_b: str,
    comp_map_a: Dict[str, float],
    comp_map_b: Dict[str, float],
    original_jaccard_pct: float
) -> List[Dict[str, Any]]:
    """
    For a single BOM pair (A, B), generate a table of hypothetical replacements:
    - Replace_In_BOM: "Replace_In_A" or "Replace_In_B"
    - Replace_Out: component to be removed from that BOM
    - Replace_In_With: component brought in from the other BOM
    - New_MatchPct: Jaccard % after that replacement
    - DeltaPct: improvement vs original_jaccard_pct (can be <= 0)
    - NewMatchedCount: |intersection|
    - NewTotalAfter: |union|
    - Direction: "A<-B" or "B<-A"

    IMPORTANT CHANGE: we NO LONGER filter out non-improving replacements.
    This guarantees we always have rows as long as both sides have unique components.
    """
    set_a = set(comp_map_a.keys())
    set_b = set(comp_map_b.keys())

    orig_inter = set_a & set_b
    orig_union = set_a | set_b

    if len(orig_union) == 0:
        base_jaccard_pct = 100.0
    else:
        base_jaccard_pct = (len(orig_inter) / len(orig_union)) * 100.0

    # Prefer the similarity coming from the main computation
    if original_jaccard_pct is not None:
        base_jaccard_pct = original_jaccard_pct

    unique_a = sorted(list(set_a - set_b))
    unique_b = sorted(list(set_b - set_a))

    rows: List[Dict[str, Any]] = []

    # Direction A<-B (modify A to be closer to B)
    for out_comp in unique_a:
        for in_comp in unique_b:
            new_set_a = (set_a - {out_comp}) | {in_comp}
            inter = new_set_a & set_b
            union = new_set_a | set_b

            if len(union) == 0:
                new_pct = 100.0
            else:
                new_pct = (len(inter) / len(union)) * 100.0

            delta = new_pct - base_jaccard_pct

            rows.append({
                "bom_a": bom_a,
                "bom_b": bom_b,
                "Replace_In_BOM": "Replace_In_A",
                "Replace_Out": out_comp,
                "Replace_In_With": in_comp,
                "New_MatchPct": round(new_pct, 2),
                "DeltaPct": round(delta, 2),
                "NewMatchedCount": int(len(inter)),
                "NewTotalAfter": int(len(union)),
                "Direction": "A<-B"
            })

    # Direction B<-A (modify B to be closer to A)
    for out_comp in unique_b:
        for in_comp in unique_a:
            new_set_b = (set_b - {out_comp}) | {in_comp}
            inter = set_a & new_set_b
            union = set_a | new_set_b

            if len(union) == 0:
                new_pct = 100.0
            else:
                new_pct = (len(inter) / len(union)) * 100.0

            delta = new_pct - base_jaccard_pct

            rows.append({
                "bom_a": bom_a,
                "bom_b": bom_b,
                "Replace_In_BOM": "Replace_In_B",
                "Replace_Out": out_comp,
                "Replace_In_With": in_comp,
                "New_MatchPct": round(new_pct, 2),
                "DeltaPct": round(delta, 2),
                "NewMatchedCount": int(len(inter)),
                "NewTotalAfter": int(len(union)),
                "Direction": "B<-A"
            })

    # Sort rows so best improvements appear first (still useful for UI)
    rows.sort(
        key=lambda r: (
            -r["DeltaPct"],          # higher improvement first
            -r["New_MatchPct"],      # then higher new match
            r["Replace_In_BOM"],     # stable ordering
            r["Replace_Out"],
            r["Replace_In_With"],
        )
    )
    return rows


def generate_component_replacement_table(
    assembly_components: Dict[str, Dict[str, float]],
    similar_pairs: List[Dict[str, Any]],
    max_pairs: int = None
) -> List[Dict[str, Any]]:
    """
    For all similar BOM pairs, generate a flat list of replacement rows
    in the format expected by the frontend:
      Replace_In_BOM, Replace_Out, Replace_In_With,
      New_MatchPct, DeltaPct, NewMatchedCount, NewTotalAfter, Direction.
    """
    rows: List[Dict[str, Any]] = []
    pairs_iter = similar_pairs
    if max_pairs is not None:
        pairs_iter = similar_pairs[:max_pairs]

    for pair in pairs_iter:
        bom_a = pair["bom_a"]
        bom_b = pair["bom_b"]
        similarity_score = pair.get("similarity_score", 0.0)
        original_jaccard_pct = similarity_score * 100.0

        comp_map_a = assembly_components.get(bom_a, {}) or {}
        comp_map_b = assembly_components.get(bom_b, {}) or {}

        pair_rows = _compute_replacement_rows_for_pair(
            bom_a=bom_a,
            bom_b=bom_b,
            comp_map_a=comp_map_a,
            comp_map_b=comp_map_b,
            original_jaccard_pct=original_jaccard_pct
        )
        rows.extend(pair_rows)

    return rows


def generate_replacement_suggestions(similar_pairs: List[Dict], limit: int = 5) -> List[Dict]:
    suggestions = []

    for pair in similar_pairs[:limit]:
        assy_a = pair["bom_a"]
        assy_b = pair["bom_b"]
        similarity = pair["similarity_score"]

        unique_a = pair.get("unique_count_a", 0)
        unique_b = pair.get("unique_count_b", 0)
        total_unique = unique_a + unique_b

        potential_savings = pair.get("common_quantity_total", pair.get("common_count", 0))

        suggestion = {
            "type": "bom_consolidation",
            "bom_a": assy_a,
            "bom_b": assy_b,
            "similarity_score": similarity,
            "suggestion": f"Consolidate {assy_a} and {assy_b} ({(similarity*100):.1f}% similar)",
            "confidence": similarity,
            "potential_savings": potential_savings,
            "details": {
                "common_components": pair.get("common_count", 0),
                "common_quantity_total": potential_savings,
                "unique_to_a": unique_a,
                "unique_to_b": unique_b
            }
        }
        suggestions.append(suggestion)

    return suggestions


def find_assembly_clusters(assemblies: List[str], similarity_matrix: Dict, threshold: float = 80.0) -> List[List[str]]:
    """Group assemblies into clusters based on similarity (uses threshold 80 by default)"""
    clusters = []
    used_assemblies = set()

    for assembly in assemblies:
        if assembly not in used_assemblies:
            cluster = [assembly]
            used_assemblies.add(assembly)

            for other_assembly in assemblies:
                if (other_assembly not in used_assemblies and
                        similarity_matrix.get(assembly, {}).get(other_assembly, 0) > threshold):
                    cluster.append(other_assembly)
                    used_assemblies.add(other_assembly)

            clusters.append(cluster)

    return clusters


def calculate_reduction_potential(clusters: List[List[str]], total_assemblies: int) -> float:
    if total_assemblies == 0:
        return 0.0

    total_reduction = 0
    for cluster in clusters:
        total_reduction += max(0, len(cluster) - 1)

    reduction_potential = (total_reduction / total_assemblies) * 100
    return round(reduction_potential, 1)


def analyze_bom_data(bom_df: pd.DataFrame, threshold: float = 70.0) -> Dict[str, Any]:
    """Main BOM analysis function (quantity-aware)"""
    print("\n=== Starting BOM Analysis ===")

    bom_df_processed = preprocess_bom_file(bom_df)

    # Filter out assembly rows for component analysis
    component_df = bom_df_processed[bom_df_processed['lev'] > 0].copy()

    assemblies = component_df['assembly_id'].unique()
    num_assemblies = len(assemblies)

    print(f"Assemblies found: {num_assemblies}")

    if num_assemblies < 2:
        print("Need at least 2 assemblies for analysis")
        return create_empty_bom_results()

    # Create component -> quantity dicts for each assembly
    assembly_components: Dict[str, Dict[str, float]] = {}
    for assembly in assemblies:
        assembly_data = component_df[component_df['assembly_id'] == assembly]
        comp_qty: Dict[str, float] = {}
        for _, r in assembly_data.iterrows():
            comp_name = str(r['component']).strip()
            qty = r.get('quantity', 0.0)
            try:
                qty = float(qty) if not pd.isna(qty) else 0.0
            except Exception:
                qty = 0.0
            comp_qty[comp_name] = comp_qty.get(comp_name, 0.0) + qty
        assembly_components[assembly] = comp_qty

    # Compute similarity
    similarity_results = compute_bom_similarity(assembly_components, threshold)

    # Generate suggestions & clusters
    replacement_suggestions = generate_replacement_suggestions(similarity_results["similar_pairs"])
    clusters = find_assembly_clusters(list(assemblies), similarity_results["similarity_matrix"])

    # NEW: generate detailed component-replacement table
    component_replacement_table = generate_component_replacement_table(
        assembly_components=assembly_components,
        similar_pairs=similarity_results["similar_pairs"],
        max_pairs=None
    )

    # Calculate statistics
    total_components = len(component_df)
    unique_components = component_df['component'].nunique()
    reduction_potential = calculate_reduction_potential(clusters, num_assemblies)

    final_results = {
        "similarity_matrix": similarity_results["similarity_matrix"],
        "similar_pairs": similarity_results["similar_pairs"],
        "replacement_suggestions": replacement_suggestions,
        "component_replacement_table": component_replacement_table,
        "bom_statistics": {
            "total_components": total_components,
            "unique_components": unique_components,
            "total_assemblies": num_assemblies,
            "total_clusters": len(clusters),
            "similar_pairs_count": len(similarity_results["similar_pairs"]),
            "reduction_potential": reduction_potential
        },
        "clusters": clusters
    }

    print(f"Analysis complete: {num_assemblies} assemblies, {len(similarity_results['similar_pairs'])} similar pairs")
    print(f"Generated {len(component_replacement_table)} component-level replacement rows")
    return final_results

