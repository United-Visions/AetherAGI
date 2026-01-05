# Supabase Migration Complete ✅

## What Changed:

### 1. **Replaced JSON File Storage with Supabase PostgreSQL**
- **Old:** `config/keys.json` (local file, not scalable)
- **New:** Supabase PostgreSQL `api_keys` table (cloud database)

### 2. **Enhanced API Key Storage with GitHub Info**

**Table Schema:**
```sql
api_keys:
  - id (UUID)
  - user_id (TEXT) - GitHub username
  - github_username (TEXT) - GitHub username
  - github_url (TEXT) - GitHub profile URL
  - key_hash (TEXT) - SHA-256 hash of API key
  - role (TEXT) - free/pro/enterprise/admin
  - created_at (TIMESTAMPTZ)
  - last_used (TIMESTAMPTZ)
  - metadata (JSONB)
  - revoked (BOOLEAN)
```

### 3. **Files Updated:**

✅ **orchestrator/supabase_client.py** - New Supabase client singleton
✅ **orchestrator/auth_manager_supabase.py** - New auth manager using Supabase
✅ **orchestrator/main_api.py** - Uses AuthManagerSupabase instead of AuthManager
✅ **frontend_flask/app.py** - Uses AuthManagerSupabase, stores GitHub info
✅ **scripts/setup_supabase_tables.sql** - SQL to create tables
✅ **scripts/setup_supabase.py** - Python setup script

### 4. **GitHub OAuth Flow Enhanced:**

Now captures and stores:
- `github_username` - Login name
- `github_url` - Profile URL (https://github.com/username)
- `github_avatar` - Avatar image URL
- `github_email` - Email address

## 🚀 Setup Instructions:

### Step 1: Install Supabase Client
```bash
cd /Users/deion/Desktop/aethermind_universal
source .venv/bin/activate
pip install supabase
```

### Step 2: Create Database Table
Go to your Supabase project:
1. Open **SQL Editor**: https://supabase.com/dashboard/project/ckjsrdwsfodwypishmsp/sql/new
2. Copy and paste the SQL from `scripts/setup_supabase_tables.sql`
3. Click **Run** to execute

### Step 3: Verify Setup
```bash
python scripts/setup_supabase.py
```

### Step 4: Restart Services
```bash
# Terminal 1 - Orchestrator
source .venv/bin/activate
python3 -m uvicorn orchestrator.main_api:app --host 0.0.0.0 --port 8000

# Terminal 2 - Flask
cd frontend_flask
source ../.venv/bin/activate
python app.py
```

## 🔐 Authentication Flow:

1. User logs in via **GitHub OAuth**
2. Flask captures: `username`, `url`, `avatar`, `email`
3. User clicks **"Create API Key"** on onboarding
4. Flask calls: `auth_mgr.generate_api_key(user_id, github_username, github_url, role=PRO)`
5. AuthManager:
   - Generates `am_live_XYZ123...`
   - Hashes with SHA-256
   - Stores hash + GitHub info in **Supabase**
6. Key shown to user (ONLY ONCE)
7. User makes requests with key in `X-Aether-Key` header
8. Backend:
   - Hashes provided key
   - Queries Supabase for match
   - Returns `user_id`, `role`, `permissions`

## 📊 Benefits:

✅ **Scalable** - Cloud database, not local files
✅ **Multi-server** - Render/Vercel can both access same DB
✅ **Automatic backups** - Supabase handles backups
✅ **Row-level security** - Built-in RLS policies
✅ **Real-time** - Can use Supabase realtime subscriptions
✅ **GitHub integration** - Stores GitHub username/URL for code activities
✅ **Better tracking** - Last used timestamps, revocation status

## 🧪 Testing:

### Create a new key:
1. Go to http://127.0.0.1:5000/onboarding
2. Login with GitHub
3. Select domain and tier
4. Click "Create API Key"
5. Key is generated and stored in Supabase

### Verify in Supabase:
1. Go to **Table Editor**: https://supabase.com/dashboard/project/ckjsrdwsfodwypishmsp/editor
2. Select `api_keys` table
3. See your new key entry with GitHub info

### Test API request:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Aether-Key: am_live_YOUR_KEY_HERE" \
  -d '{
    "model": "aethermind-v1",
    "user": "test_user",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🔧 Environment Variables:

Make sure these are set in `.env`:
```bash
SB_URL=https://ckjsrdwsfodwypishmsp.supabase.co
SB_SECRET_KEY=sb_secret_nJ2MVzxv9wjdMzvGD-yCHg_ALgyYtjI
SB_DB_PASSWORD=Mc1417182613.
SB_POSTGRESQL_URL=postgresql://postgres:Mc1417182613.@db.ckjsrdwsfodwypishmsp.supabase.co:5432/postgres
```

## ⚠️ Important Notes:

1. **No more keys.json** - Old file is no longer used
2. **GitHub required** - GitHub OAuth is required for full AGI features (code activities)
3. **Service key** - Use `SB_SECRET_KEY` (not anon key) for backend operations
4. **Rate limiting** - Still enforced: FREE=100/min, PRO=1000/min
5. **SDK compatible** - Works with existing SDK, just uses Supabase backend

## 🎯 Next Steps:

1. ✅ Run `setup_supabase.py` to create tables
2. ✅ Test key generation on onboarding
3. ✅ Test API requests with new keys
4. Deploy to Render/Vercel with Supabase env vars
