# DeepRPS Deployment Guide

## Cloud Deployment Options (Free Tier)

### 1. Railway (Recommended for Full-Stack)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway new
railway link
railway up
```

**Environment Variables:**
```env
STATE_DIR=/app/data
DATABASE_PATH=/app/data/gamebrain.db
STATIC_PATH=/app/static
```

### 2. Render (Alternative)
- Connect GitHub repository
- Choose "Web Service"
- Build Command: `docker build -t deeprps .`
- Start Command: `python -m uvicorn gamebrain.server.main:app --host 0.0.0.0 --port $PORT`

### 3. Vercel (Frontend) + Railway (Backend)
Frontend on Vercel:
```bash
cd FE
vercel --prod
```

Backend on Railway with database.

## Database Options

### Option 1: SQLite (Default)
- No setup required
- File-based storage
- Perfect for small-scale deployment
- Persistent across container restarts with volume mounting

### Option 2: Supabase (Cloud Database)
```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
```

SQL Schema for Supabase:
```sql
CREATE TABLE user_states (
    user_id TEXT PRIMARY KEY,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_states_updated_at 
    BEFORE UPDATE ON user_states 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### Option 3: MongoDB Atlas (Alternative)
For larger scale, can implement MongoDB adapter.

## Local Development
```bash
# Backend
cd AI/gamebrain
python -m server.main

# Frontend
cd FE  
npm run dev
```

## Production Considerations

1. **Data Persistence**: Mount volume for SQLite or use cloud database
2. **Environment Variables**: Set proper STATE_DIR and database credentials
3. **CORS**: Configure properly for your domain
4. **Scaling**: Consider Redis for session storage if needed
5. **Monitoring**: Add logging and health checks

## Cost Analysis
- **Railway**: $5/month for persistent storage, $0 for hobby tier
- **Render**: Free tier with 512MB RAM, sleeps after 15min
- **Supabase**: Free tier: 500MB database, 50MB file storage
- **Vercel**: Free tier for frontend hosting

## Recommended Setup for Production:
1. **Frontend**: Vercel (free)
2. **Backend**: Railway ($5/month)  
3. **Database**: SQLite on Railway volume or Supabase free tier
4. **Total Cost**: $0-5/month depending on usage