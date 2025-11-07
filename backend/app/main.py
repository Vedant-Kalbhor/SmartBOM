from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
import uuid
import re
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

app = FastAPI(title="BOM Optimization Tool", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# In-memory storage
weldment_data = {}
bom_data = {}
analysis_results = {}

def generate_file_id():
    return str(uuid.uuid4())

def clean_column_name(column_name: str) -> str:
    """Clean column names for consistency"""
    if pd.isna(column_name) or column_name is None:
        return "unknown"
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', str(column_name).lower())
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

# BOM Analysis Functions
def preprocess_bom_file(bom_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess BOM file to create proper assembly_id from hierarchical structure"""
    # Normalize column names to lowercase
    bom_df.columns = bom_df.columns.str.lower().str.strip()
    
    # Ensure required columns exist
    if 'component' not in bom_df.columns:
        if 'component' in bom_df.columns:
            bom_df.rename(columns={'component': 'component'}, inplace=True)
    
    if 'lev' not in bom_df.columns:
        if 'lev' in bom_df.columns:
            bom_df.rename(columns={'lev': 'lev'}, inplace=True)
    
    # Create assembly_id based on Lev hierarchy
    current_assembly = None
    assembly_ids = []
    
    for idx, row in bom_df.iterrows():
        lev = row['lev']
        component = row['component']
        
        if lev == 0:
            current_assembly = component
            assembly_ids.append(current_assembly)
        else:
            if current_assembly is None:
                current_assembly = f"ASSY_{idx}"
            assembly_ids.append(current_assembly)
    
    bom_df['assembly_id'] = assembly_ids
    bom_df['is_assembly'] = bom_df['lev'] == 0
    
    print(f"Identified {bom_df[bom_df['is_assembly']]['assembly_id'].nunique()} unique assemblies")
    return bom_df

def compute_bom_similarity(assembly_components: Dict[str, set]) -> Dict[str, Any]:
    """Compute BOM similarity between assemblies"""
    assemblies = list(assembly_components.keys())
    num_assemblies = len(assemblies)
    
    if num_assemblies < 2:
        return create_empty_bom_results()
    
    similarity_matrix = {}
    similar_pairs = []
    
    for i, assy1 in enumerate(assemblies):
        similarity_matrix[assy1] = {}
        set1 = assembly_components[assy1]
        
        for j, assy2 in enumerate(assemblies):
            set2 = assembly_components[assy2]
            
            # Read the file again without header to see all data
            df_raw = pd.read_excel(file_path, header=None)
            print("Raw data shape:", df_raw.shape)
            
            similarity_matrix[assy1][assy2] = round(similarity, 2)
            
            if i < j and similarity > 70:
                common_components = list(set1.intersection(set2))
                unique_to_assy1 = list(set1 - set2)
                unique_to_assy2 = list(set2 - set1)
                
                similar_pairs.append({
                    "bom_a": assy1,
                    "bom_b": assy2,
                    "similarity_score": round(similarity / 100, 4),
                    "common_components": common_components[:10],  # Limit for serialization
                    "unique_components_a": unique_to_assy1[:10],
                    "unique_components_b": unique_to_assy2[:10],
                    "common_count": len(common_components),
                    "unique_count_a": len(unique_to_assy1),
                    "unique_count_b": len(unique_to_assy2)
                })
    
    return {
        "similarity_matrix": similarity_matrix,
        "similar_pairs": similar_pairs
    }

def generate_replacement_suggestions(similar_pairs: List[Dict]) -> List[Dict]:
    """Generate replacement suggestions based on similar pairs"""
    suggestions = []
    
    for pair in similar_pairs[:5]:
        assy_a = pair["bom_a"]
        assy_b = pair["bom_b"]
        similarity = pair["similarity_score"]
        
        unique_a = pair["unique_count_a"]
        unique_b = pair["unique_count_b"]
        total_unique = unique_a + unique_b
        
        potential_savings = min(total_unique * 10, 100)
        
        suggestion = {
            "type": "bom_consolidation",
            "bom_a": assy_a,
            "bom_b": assy_b,
            "similarity_score": similarity,
            "suggestion": f"Consolidate {assy_a} and {assy_b} ({(similarity*100):.1f}% similar)",
            "confidence": similarity,
            "potential_savings": potential_savings,
            "details": {
                "common_components": pair["common_count"],
                "unique_to_a": unique_a,
                "unique_to_b": unique_b
            }
        }
        suggestions.append(suggestion)
    
    return suggestions

def find_assembly_clusters(assemblies: List[str], similarity_matrix: Dict) -> List[List[str]]:
    """Group assemblies into clusters based on similarity"""
    clusters = []
    used_assemblies = set()
    
    for assembly in assemblies:
        if assembly not in used_assemblies:
            cluster = [assembly]
            used_assemblies.add(assembly)
            
            for other_assembly in assemblies:
                if (other_assembly not in used_assemblies and 
                    similarity_matrix.get(assembly, {}).get(other_assembly, 0) > 80):
                    cluster.append(other_assembly)
                    used_assemblies.add(other_assembly)
            
            clusters.append(cluster)
    
    return clusters

def calculate_reduction_potential(clusters: List[List[str]], total_assemblies: int) -> float:
    """Calculate potential reduction in number of assemblies"""
    if total_assemblies == 0:
        return 0.0
    
    total_reduction = 0
    for cluster in clusters:
        total_reduction += max(0, len(cluster) - 1)
    
    reduction_potential = (total_reduction / total_assemblies) * 100
    return round(reduction_potential, 1)

def create_empty_bom_results() -> Dict[str, Any]:
    """Create empty results structure"""
    return {
        "similarity_matrix": {},
        "similar_pairs": [],
        "replacement_suggestions": [],
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

def analyze_bom_data(bom_df: pd.DataFrame) -> Dict[str, Any]:
    """Main BOM analysis function"""
    print("\n=== Starting BOM Analysis ===")
    
    # Preprocess the BOM data
    bom_df_processed = preprocess_bom_file(bom_df)
    
    # Filter out assembly rows for component analysis
    component_df = bom_df_processed[bom_df_processed['lev'] > 0].copy()
    
    # Get unique assemblies
    assemblies = component_df['assembly_id'].unique()
    num_assemblies = len(assemblies)
    
    print(f"Assemblies found: {num_assemblies}")
    
    if num_assemblies < 2:
        print("Need at least 2 assemblies for analysis")
        return create_empty_bom_results()
    
    # Create component sets for each assembly
    assembly_components = {}
    for assembly in assemblies:
        assembly_data = component_df[component_df['assembly_id'] == assembly]
        components = set(assembly_data['component'].tolist())
        assembly_components[assembly] = components
    
    # Compute similarity
    similarity_results = compute_bom_similarity(assembly_components)
    
    # Generate additional results
    replacement_suggestions = generate_replacement_suggestions(similarity_results["similar_pairs"])
    clusters = find_assembly_clusters(list(assemblies), similarity_results["similarity_matrix"])
    
    # Calculate statistics
    total_components = len(component_df)
    unique_components = component_df['component'].nunique()
    reduction_potential = calculate_reduction_potential(clusters, num_assemblies)
    
    # Build final results - AVOID CIRCULAR REFERENCES
    final_results = {
        "similarity_matrix": similarity_results["similarity_matrix"],
        "similar_pairs": similarity_results["similar_pairs"],
        "replacement_suggestions": replacement_suggestions,
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
    return final_results

# File parsing functions
def parse_weldment_excel(file_path: str) -> pd.DataFrame:
    """Parse weldment Excel file"""
    try:
        df = pd.read_excel(file_path)
        df.columns = [clean_column_name(col) for col in df.columns]
        return df
    except Exception as e:
        print(f"Error parsing Excel file: {str(e)}")
        raise

def validate_weldment_columns(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame contains all required weldment columns (flexible matching)"""
    print("Validating weldment columns...")
    print("Columns received:", df.columns.tolist())

    # Define required logical column keys and their expected keywords
    required_patterns = {
        "assy_pn": ["assy", "pn"],
        "total_height_of_packed_tower_mm": ["total", "height", "packed", "tower"],
        "packed_tower_outer_dia_mm": ["packed", "tower", "outer", "dia"],
        "packed_tower_inner_dia_mm": ["packed", "tower", "inner", "dia"],
        "upper_flange_outer_dia_mm": ["upper", "flange", "outer", "dia"],
        "upper_flange_inner_dia_mm": ["upper", "flange", "inner", "dia"],
        "lower_flange_outer_dia_mm": ["lower", "flange", "outer", "dia"],
        "spray_nozzle_center_distance": ["spray", "nozzle", "center", "distance"],
        "spray_nozzle_id": ["spray", "nozzle", "id"],
        "support_ring_height_from_bottom": ["support", "ring", "height"],
        "support_ring_id": ["support", "ring", "id"]
    }

    # Normalize and clean column names for comparison
    cleaned_cols = [clean_column_name(col) for col in df.columns]

    def matches_pattern(col: str, keywords: list[str]) -> bool:
        return all(k in col for k in keywords)

    missing_columns = []
    for key, keywords in required_patterns.items():
        if not any(matches_pattern(col, keywords) for col in cleaned_cols):
            missing_columns.append(key)

    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {cleaned_cols}")
        return False

    print("✅ All required columns are present")
    return True


def validate_weldment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean weldment dimension data"""
    print("Validating weldment data...")
    print("Input columns:", df.columns.tolist())
    print("Data shape:", df.shape)

    # Clean all column names once for consistency
    df.columns = [clean_column_name(col) for col in df.columns]

    # Check if required columns exist (using flexible validation)
    if not validate_weldment_columns(df):
        raise ValueError("Weldment file is missing required columns. Please ensure the file contains all 11 required columns.")

    # Remove empty rows
    df = df.dropna(how='all')
    if len(df) == 0:
        raise ValueError("No data found in the file")

    # Ensure 'assy_pn' column exists
    if 'assy_pn' not in df.columns:
        raise ValueError("Missing 'Assy PN' column after cleaning")

    # Drop rows without Assy PN
    df = df.dropna(subset=['assy_pn'])
    df['assy_pn'] = df['assy_pn'].astype(str).str.strip()

    # Convert all other numeric columns
    numeric_columns = [col for col in df.columns if col != 'assy_pn']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"✅ Weldment data validated successfully. Shape: {df.shape}")
    print("Final columns:", df.columns.tolist())
    return df


def validate_bom_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean BOM data"""
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Find required columns
    component_col = None
    lev_col = None
    quantity_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'component' in col_lower:
            component_col = col
        elif 'lev' in col_lower:
            lev_col = col
        elif 'quantity' in col_lower:
            quantity_col = col
    
    # Rename columns
    if component_col:
        df = df.rename(columns={component_col: 'component'})
    if lev_col:
        df = df.rename(columns={lev_col: 'lev'})
    if quantity_col:
        df = df.rename(columns={quantity_col: 'quantity'})
    
    # Check required columns
    missing_columns = []
    if 'component' not in df.columns:
        missing_columns.append('component')
    if 'lev' not in df.columns:
        missing_columns.append('lev')
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean data
    df = df.dropna(subset=['component'])
    df['lev'] = pd.to_numeric(df['lev'], errors='coerce')
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    
    # Add assembly ID for grouping if not present
    if 'assembly_id' not in df.columns:
        # Try to infer assembly from component structure
        if 'assy' in df.columns.tolist():
            df = df.rename(columns={'assy': 'assembly_id'})
        else:
            # Group by level 0 components as assemblies
            level_0_components = df[df['lev'] == 0]['component'].unique()
            if len(level_0_components) > 0:
                df['assembly_id'] = df['component'].apply(
                    lambda x: next((comp for comp in level_0_components if str(comp) in str(x)), 'default_assembly')
                )
            else:
                df['assembly_id'] = 'default_assembly'
    
    print(f"BOM data validated. Records: {len(df)}")
    print("Assembly IDs:", df['assembly_id'].unique())
    return df

# API Endpoints
@app.post("/upload/weldments/")
async def upload_weldments(file: UploadFile = File(...)):
    """Upload weldment dimensions file"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = parse_weldment_excel(file_path)
        
        validated_data = validate_weldment_data(df)
        
        file_id = generate_file_id()
        weldment_data[file_id] = {
            "filename": file.filename,
            "data": validated_data.to_dict('records'),
            "columns": validated_data.columns.tolist(),
            "record_count": len(validated_data),
            "dataframe": validated_data  # Store the actual DataFrame for analysis
        }
        
        return {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "record_count": len(validated_data),
            "columns": validated_data.columns.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/upload/boms/")
async def upload_boms(file: UploadFile = File(...)):
    """Upload BOM file"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            try:
                xl = pd.ExcelFile(file_path)
                sheet_names = xl.sheet_names
                bom_sheets = [name for name in sheet_names if 'bom' in name.lower() or 'assy' in name.lower()]
                if bom_sheets:
                    df = pd.read_excel(file_path, sheet_name=bom_sheets[0])
                else:
                    df = pd.read_excel(file_path)
            except:
                df = pd.read_excel(file_path)
        
        validated_data = validate_bom_data(df)
        
        file_id = generate_file_id()
        bom_data[file_id] = {
            "filename": file.filename,
            "data": validated_data.to_dict('records'),
            "columns": validated_data.columns.tolist(),
            "record_count": len(validated_data),
            "dataframe": validated_data  # Store the actual DataFrame for analysis
        }
        
        return {
            "message": "BOM file uploaded successfully",
            "file_id": file_id,
            "record_count": len(validated_data),
            "columns": validated_data.columns.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.get("/files/weldments/")
async def get_weldment_files():
    """Get list of uploaded weldment files"""
    return [
        {
            "file_id": fid, 
            "filename": data["filename"], 
            "record_count": data["record_count"],
            "columns": data["columns"]
        }
        for fid, data in weldment_data.items()
    ]

@app.get("/files/boms/")
async def get_bom_files():
    """Get list of uploaded BOM files"""
    return [
        {
            "file_id": fid, 
            "filename": data["filename"], 
            "record_count": data["record_count"],
            "columns": data["columns"]
        }
        for fid, data in bom_data.items()
    ]

@app.get("/weldment-data/{file_id}")
async def get_weldment_data(file_id: str):
    """Get actual weldment data for visualization"""
    if file_id not in weldment_data:
        raise HTTPException(status_code=404, detail="Weldment file not found")
    
    return {
        "data": weldment_data[file_id]["data"],
        "columns": weldment_data[file_id]["columns"]
    }

@app.post("/analyze/dimensional-clustering/")
async def analyze_dimensional_clustering(request: dict):
    """Perform dimensional clustering analysis with real data"""
    try:
        weldment_file_id = request.get('weldment_file_id')
        clustering_method = request.get('clustering_method', 'kmeans')
        n_clusters = request.get('n_clusters')
        tolerance = request.get('tolerance', 0.1)
        
        if weldment_file_id not in weldment_data:
            raise HTTPException(status_code=404, detail="Weldment file not found")
        
        # Get the actual DataFrame
        df = weldment_data[weldment_file_id]["dataframe"]
        
        print("Clustering data columns:", df.columns.tolist())
        print("Clustering data shape:", df.shape)
        print("Data sample:", df.head())
        
        # Select numeric columns for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print("Numeric columns for clustering:", numeric_cols)
        
        if len(numeric_cols) < 2:
            cluster_results = []
            visualization_data = []
        else:
            # Prepare features for clustering
            features = df[numeric_cols].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Validate and convert n_clusters parameter
            if n_clusters is not None:
                try:
                    # Convert to integer and validate range
                    n_clusters = int(n_clusters)
                    max_possible_clusters = len(df)
                    if n_clusters < 2:
                        n_clusters = 2
                    elif n_clusters > max_possible_clusters:
                        n_clusters = max_possible_clusters
                        print(f"Warning: n_clusters reduced to {max_possible_clusters} (number of data points)")
                except (ValueError, TypeError):
                    print(f"Invalid n_clusters value: {n_clusters}, using automatic detection")
                    n_clusters = None
            
            # Determine number of clusters if not provided or invalid
            if n_clusters is None:
                n_clusters = min(5, max(2, len(df) // 3))
            
            print(f"Using n_clusters: {n_clusters}")
            
            # Perform clustering based on method
            if clustering_method == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)
            elif clustering_method == 'hierarchical':
                Z = linkage(scaled_features, method='ward')
                clusters = fcluster(Z, n_clusters, criterion='maxclust') - 1
            elif clustering_method == 'dbscan':
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                clusters = dbscan.fit_predict(scaled_features)
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            df['cluster'] = clusters
            
            cluster_results = []
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise in DBSCAN
                    continue
                    
                cluster_data = df[df['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    # Find representative (closest to centroid)
                    if clustering_method == 'kmeans':
                        centroid = kmeans.cluster_centers_[cluster_id]
                        cluster_scaled = scaler.transform(cluster_data[numeric_cols].fillna(0))
                        distances = np.linalg.norm(cluster_scaled - centroid, axis=1)
                        representative_idx = distances.argmin()
                    else:
                        representative_idx = 0
                    
                    representative = cluster_data.iloc[representative_idx]['assy_pn']
                    
                    cluster_results.append({
                        "cluster_id": int(cluster_id),
                        "member_count": len(cluster_data),
                        "members": cluster_data['assy_pn'].tolist(),
                        "representative": representative,
                        "reduction_potential": max(0, len(cluster_data) - 1) / len(cluster_data) if len(cluster_data) > 0 else 0
                    })
            
            # Prepare visualization data with actual values
            visualization_data = []
            for _, row in df.iterrows():
                point_data = {
                    'assy_pn': row['assy_pn'],
                    'cluster': int(row['cluster'])
                }
                # Add all numeric columns for visualization
                for col in numeric_cols:
                    point_data[col] = row[col] if not pd.isna(row[col]) else 0
                visualization_data.append(point_data)
        
        analysis_id = generate_file_id()
        
        # Store complete analysis results
        analysis_results[analysis_id] = {
            "type": "clustering",
            "clustering": {
                "clusters": cluster_results,
                "metrics": {
                    "n_clusters": len(cluster_results),
                    "n_samples": len(df),
                    "silhouette_score": silhouette_score(scaled_features, clusters) if len(unique_clusters) > 1 else 0
                },
                "visualization_data": visualization_data,
                "numeric_columns": numeric_cols
            },
            "bom_analysis": {
                "similar_pairs": [],
                "replacement_suggestions": []
            }
        }
        
        return {
            "analysis_id": analysis_id,
            "clustering_result": analysis_results[analysis_id]["clustering"],
            "bom_analysis_result": analysis_results[analysis_id]["bom_analysis"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.post("/analyze/bom-similarity/")
async def analyze_bom_similarity(request: dict):
    """Perform BOM similarity analysis with real data"""
    try:
        bom_file_id = request.get('bom_file_id')
        similarity_method = request.get('similarity_method', 'jaccard')
        threshold = request.get('threshold', 0.8)
        
        if bom_file_id not in bom_data:
            raise HTTPException(status_code=404, detail="BOM file not found")
        
        # Get the actual DataFrame
        df = bom_data[bom_file_id]["dataframe"]
        
        print("BOM data for analysis:")
        print(df.head())
        print("Assemblies:", df['assembly_id'].unique())
        
        # Group by assembly and create BOM representations
        assemblies = df['assembly_id'].unique()
        bom_vectors = {}
        
        for assembly in assemblies:
            assembly_bom = df[df['assembly_id'] == assembly]
            # Create component-frequency dictionary
            component_freq = {}
            for _, row in assembly_bom.iterrows():
                component = str(row['component'])
                quantity = row.get('quantity', 1)
                component_freq[component] = component_freq.get(component, 0) + quantity
            bom_vectors[assembly] = component_freq
        
        print("BOM vectors created for assemblies:", list(bom_vectors.keys()))
        
        # Calculate similarity matrix
        similarity_matrix = {}
        similar_pairs = []
        
        all_components = set()
        for bom in bom_vectors.values():
            all_components.update(bom.keys())
        
        for i, assembly_a in enumerate(assemblies):
            similarity_matrix[assembly_a] = {}
            components_a = set(bom_vectors[assembly_a].keys())
            
            for j, assembly_b in enumerate(assemblies):
                if i != j:
                    components_b = set(bom_vectors[assembly_b].keys())
                    
                    # Calculate Jaccard similarity
                    intersection = len(components_a.intersection(components_b))
                    union = len(components_a.union(components_b))
                    
                    similarity = intersection / union if union > 0 else 0
                    similarity_matrix[assembly_a][assembly_b] = similarity
                    
                    if similarity >= threshold:
                        similar_pairs.append({
                            "bom_a": assembly_a,
                            "bom_b": assembly_b,
                            "similarity_score": similarity,
                            "common_components": intersection,
                            "unique_components_a": list(components_a - components_b),
                            "unique_components_b": list(components_b - components_a)
                        })
        
        # Generate replacement suggestions
        replacement_suggestions = []
        for pair in similar_pairs[:5]:  # Limit to top 5
            replacement_suggestions.append({
                "type": "bom_consolidation",
                "bom_a": pair["bom_a"],
                "bom_b": pair["bom_b"],
                "similarity_score": pair["similarity_score"],
                "suggestion": f"Consider consolidating {pair['bom_a']} and {pair['bom_b']}",
                "confidence": pair["similarity_score"],
                "potential_savings": len(pair["unique_components_a"]) + len(pair["unique_components_b"])
            })
        
        analysis_id = generate_file_id()
        
        # Store complete analysis results
        analysis_results[analysis_id] = {
            "type": "bom_analysis",
            "clustering": {
                "clusters": [],
                "metrics": {
                    "n_clusters": 0,
                    "n_samples": 0,
                    "silhouette_score": 0
                },
                "visualization_data": [],
                "numeric_columns": []
            },
            "bom_analysis": {
                "similarity_matrix": similarity_matrix,
                "similar_pairs": similar_pairs,
                "replacement_suggestions": replacement_suggestions,
                "bom_statistics": {
                    "total_components": len(df),
                    "unique_components": df['component'].nunique(),
                    "total_assemblies": len(assemblies),
                    "avg_components_per_assembly": len(df) / len(assemblies) if len(assemblies) > 0 else 0
                }
            }
        }
        
        analysis_results[analysis_id] = stored_results
        
        return {
            "analysis_id": analysis_id,
            "clustering_result": analysis_results[analysis_id]["clustering"],
            "bom_analysis_result": analysis_results[analysis_id]["bom_analysis"]
        }
    
    except Exception as e:
        print(f"BOM analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"BOM analysis failed: {str(e)}")

@app.get("/analysis/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get analysis results by ID"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/")
async def root():
    return {"message": "BOM Optimization Tool API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)