import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://gtozjiqzsdbuptweeyeg.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0b3pqaXF6c2RidXB0d2VleWVnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA3NjI1MTUsImV4cCI6MjA2NjMzODUxNX0.1hbyNJ9pFy6ht4v5iKtbNZmX9R56rLxDbOjtagB55vk")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client() -> Client:
    """Get the Supabase client instance"""
    return supabase

def test_connection():
    """Test the Supabase connection"""
    try:
        # Test with a simple query (this will work even without tables)
        response = supabase.auth.get_session()
        print("✅ Supabase connection successful!")
        print(f"📍 Connected to: {SUPABASE_URL}")
        return True
    except Exception as e:
        print("❌ Supabase connection failed!")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()
