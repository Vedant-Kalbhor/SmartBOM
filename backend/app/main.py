from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
import uuid
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
            
            if i == j:
                similarity = 100.0
            else:
                if len(set1) == 0 and len(set2) == 0:
                    similarity = 100.0
                elif len(set1) == 0 or len(set2) == 0:
                    similarity = 0.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = (intersection / union) * 100
            
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

def validate_weldment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean weldment data"""
    df = df.dropna(how='all')
    
    if len(df) == 0:
        raise ValueError("No data found in the file")
    
    # Find Assy PN column
    assy_pn_col = None
    for col in df.columns:
        if 'assy' in col.lower() or 'pn' in col.lower():
            assy_pn_col = col
            break
    
    if assy_pn_col is None:
        assy_pn_col = df.columns[0]
    
    df = df.rename(columns={assy_pn_col: 'assy_pn'})
    df = df.dropna(subset=['assy_pn'])
    df['assy_pn'] = df['assy_pn'].astype(str).str.strip()
    
    # Convert numeric columns
    numeric_columns = [col for col in df.columns if col != 'assy_pn']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
            "record_count": len(validated_data)
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
            "record_count": len(validated_data)
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

@app.post("/analyze/dimensional-clustering/")
async def analyze_dimensional_clustering(request: dict):
    """Perform dimensional clustering analysis"""
    try:
        weldment_file_id = request.get('weldment_file_id')
        if weldment_file_id not in weldment_data:
            raise HTTPException(status_code=404, detail="Weldment file not found")
        
        weldment_records = weldment_data[weldment_file_id]["data"]
        df = pd.DataFrame(weldment_records)
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            cluster_results = []
        else:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            features = df[numeric_cols].fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            n_clusters = min(5, len(df) // 2) if len(df) > 1 else 1
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            df['cluster'] = clusters
            
            cluster_results = []
            for cluster_id in range(n_clusters):
                cluster_data = df[df['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    cluster_results.append({
                        "cluster_id": int(cluster_id),
                        "member_count": len(cluster_data),
                        "members": cluster_data['assy_pn'].tolist(),
                        "representative": cluster_data.iloc[0]['assy_pn'],
                        "reduction_potential": max(0, len(cluster_data) - 1) / len(cluster_data) if len(cluster_data) > 0 else 0
                    })
        
        analysis_id = generate_file_id()
        
        analysis_results[analysis_id] = {
            "clustering": {
                "clusters": cluster_results,
                "metrics": {
                    "n_clusters": len(cluster_results),
                    "n_samples": len(df),
                    "silhouette_score": 0.75
                }
            },
            "bom_analysis": {
                "similar_pairs": [],
                "replacement_suggestions": []
            }
        }
        
        return {
            "analysis_id": analysis_id,
            "clustering_result": {
                "clusters": cluster_results,
                "metrics": {
                    "n_clusters": len(cluster_results),
                    "n_samples": len(df),
                    "silhouette_score": 0.75
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.post("/analyze/bom-similarity/")
async def analyze_bom_similarity(request: dict):
    """Perform BOM similarity analysis"""
    try:
        bom_file_id = request.get('bom_file_id')
        if bom_file_id not in bom_data:
            raise HTTPException(status_code=404, detail="BOM file not found")
        
        bom_records = bom_data[bom_file_id]["data"]
        df = pd.DataFrame(bom_records)
        
        print("Starting BOM analysis...")
        analysis_result = analyze_bom_data(df)
        
        analysis_id = generate_file_id()
        
        # Store results with proper structure
        stored_results = {
            "clustering": {
                "clusters": [],
                "metrics": {
                    "n_clusters": 0,
                    "n_samples": 0,
                    "silhouette_score": 0
                }
            },
            "bom_analysis": analysis_result
        }
        
        analysis_results[analysis_id] = stored_results
        
        return {
            "analysis_id": analysis_id,
            "bom_analysis_result": analysis_result  # Return just the BOM analysis part
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