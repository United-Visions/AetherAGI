
import os
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from dotenv import load_dotenv
load_dotenv()

from orchestrator.auth_manager_supabase import AuthManagerSupabase, UserRole

def main():
    auth = AuthManagerSupabase()
    # Create an admin key for local benchmarking
    key = auth.generate_api_key(
        user_id="local_tester", 
        github_username="local_tester", 
        role=UserRole.ADMIN
    )
    print(key)

if __name__ == "__main__":
    main()
