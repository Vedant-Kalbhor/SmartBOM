from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB Connection URL
MONGO_URL = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bom_optimization")  # fallback default

if not MONGO_URL:
    raise ValueError("MONGO_URI is missing in your .env file")

# Create a global Mongo client (singleton)
client = MongoClient(MONGO_URL)

# Select database (use DB_NAME from env or default)
db = client[DB_NAME]

# Expose collections
analysis_collection = db["analysis_results"]
bom_files_collection = db["bom_files"]
users_collection = db["users"]

# Ensure unique email for users
users_collection.create_index("email", unique=True)
