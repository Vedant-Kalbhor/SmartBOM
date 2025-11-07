from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from typing import List, Dict, Any
import json
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

def parse_weldment_excel(file_path: str) -> pd.DataFrame:
    """Parse weldment Excel file with complex headers"""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        print("Original columns:", df.columns.tolist())
        print("First few rows of data:")
        print(df.head())
        
        # If we have Unnamed columns, try to find the actual header row
        if all('unnamed' in str(col).lower() for col in df.columns):
            print("Detected Unnamed columns, trying to find header...")
            
            # Read the file again without header to see all data
            df_raw = pd.read_excel(file_path, header=None)
            print("Raw data shape:", df_raw.shape)
            
            # Look for the row that contains 'Assy PN' - this should be our header
            header_row_idx = None
            for idx in range(min(10, len(df_raw))):
                row_values = df_raw.iloc[idx].values
                if 'Assy PN' in str(row_values[0]):
                    header_row_idx = idx
                    break
            
            if header_row_idx is not None:
                print(f"Found header at row {header_row_idx}")
                # Read with the correct header row
                df = pd.read_excel(file_path, header=header_row_idx)
                print("Columns after header detection:", df.columns.tolist())
            else:
                # If we can't find the header, use the first row and create meaningful column names
                print("Could not find header row, using first row as data")
                df = pd.read_excel(file_path, header=0)
                # Create meaningful column names based on position
                new_columns = [
                    'assy_pn',
                    'total_height_mm', 
                    'packed_tower_outer_dia_mm',
                    'packed_tower_inner_dia_mm',
                    'upper_flange_outer_dia_mm',
                    'upper_flange_inner_dia_mm',
                    'lower_flange_outer_dia_mm',
                    'spray_nozzle_center_distance',
                    'spray_nozzle_id',
                    'support_ring_height',
                    'support_ring_id'
                ]
                # Use as many columns as we have
                df.columns = new_columns[:len(df.columns)]
        
        # Clean the column names
        df.columns = [clean_column_name(col) for col in df.columns]
        print("Cleaned columns:", df.columns.tolist())
        
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
    print("Validating BOM data...")
    print("BOM columns:", df.columns.tolist())
    
    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Look for required columns with flexible matching
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
    
    # If we found the columns, rename them for consistency
    if component_col:
        df = df.rename(columns={component_col: 'component'})
    if lev_col:
        df = df.rename(columns={lev_col: 'lev'})
    if quantity_col:
        df = df.rename(columns={quantity_col: 'quantity'})
    
    # Check if we have the required columns
    missing_columns = []
    if 'component' not in df.columns:
        missing_columns.append('component')
    if 'lev' not in df.columns:
        missing_columns.append('lev')
    if 'quantity' not in df.columns:
        missing_columns.append('quantity')
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")
    
    # Remove empty rows
    df = df.dropna(subset=['component'])
    
    # Convert numeric columns
    df['lev'] = pd.to_numeric(df['lev'], errors='coerce')
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

@app.post("/upload/weldments/")
async def upload_weldments(file: UploadFile = File(...)):
    """Upload weldment dimensions file"""
    try:
        print(f"Processing weldment file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read and parse the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = parse_weldment_excel(file_path)
        
        # Validate and clean the data
        validated_data = validate_weldment_data(df)
        
        # Store the data
        file_id = generate_file_id()
        weldment_data[file_id] = {
            "filename": file.filename,
            "data": validated_data.to_dict('records'),
            "file_path": file_path,
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
        print(f"Error processing weldment file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/upload/boms/")
async def upload_boms(file: UploadFile = File(...)):
    """Upload BOM file"""
    try:
        print(f"Processing BOM file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Try to read specific sheet for BOM data
            try:
                xl = pd.ExcelFile(file_path)
                sheet_names = xl.sheet_names
                print(f"Available sheets: {sheet_names}")
                
                # Look for sheets that might contain BOM data
                bom_sheets = [name for name in sheet_names if 'bom' in name.lower() or 'assy' in name.lower()]
                if bom_sheets:
                    df = pd.read_excel(file_path, sheet_name=bom_sheets[0])
                    print(f"Using sheet: {bom_sheets[0]}")
                else:
                    df = pd.read_excel(file_path)  # Use first sheet
            except Exception as e:
                df = pd.read_excel(file_path)  # Fallback to first sheet
        
        print(f"Original BOM columns: {df.columns.tolist()}")
        print(f"BOM Data shape: {df.shape}")
        
        # Validate and clean the data
        validated_data = validate_bom_data(df)
        
        # Store the data
        file_id = generate_file_id()
        bom_data[file_id] = {
            "filename": file.filename,
            "data": validated_data.to_dict('records'),
            "file_path": file_path,
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
        print(f"Error processing BOM file: {str(e)}")
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
            # Return empty but properly structured response
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
            
            # Prepare cluster results
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
        print(f"Clustering analysis failed: {str(e)}")
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
        
        return {
            "analysis_id": analysis_id,
            "clustering_result": analysis_results[analysis_id]["clustering"],
            "bom_analysis_result": analysis_results[analysis_id]["bom_analysis"]
        }
    
    except Exception as e:
        print(f"BOM analysis failed: {str(e)}")
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
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)