# 🚀 FREE DEPLOYMENT GUIDE - DeepRPS

## Bước 1: Push code lên GitHub

### 1.1 Tạo GitHub repository
1. Đi tới [github.com](https://github.com) và đăng nhập
2. Click "New repository"
3. Đặt tên: `DeepRPS` 
4. Chọn Public (để dùng free tier)
5. Click "Create repository"

### 1.2 Push code từ local
```bash
cd c:\Users\hongo\DeepRPS

# Initialize git if not done
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - DeepRPS game with AI"

# Add remote origin (thay YOUR_USERNAME bằng GitHub username của bạn)
git remote add origin https://github.com/YOUR_USERNAME/DeepRPS.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Bước 2: Setup Supabase Database (FREE)

### 2.1 Tạo Supabase project
1. Đi tới [supabase.com](https://supabase.com)
2. Click "Start your project" → "Sign up" (dùng GitHub account)
3. Click "New project"
4. Chọn Organization: Personal
5. Đặt tên project: `deeprps`
6. Tạo Database Password (lưu lại!)
7. Chọn Region: gần Việt Nam nhất (Singapore)
8. Click "Create new project"

### 2.2 Tạo bảng user_states
1. Đi tới SQL Editor trong Supabase dashboard
2. Paste và chạy query này:

```sql
CREATE TABLE user_states (
    user_id TEXT PRIMARY KEY,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE user_states ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (for simplicity)
CREATE POLICY "Allow all operations" ON user_states FOR ALL USING (true);

-- Trigger to update updated_at
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

### 2.3 Lấy connection details
1. Đi tới Settings → API
2. Copy các thông tin này:
   - Project URL
   - anon/public key

## Bước 3: Deploy lên Vercel (FREE)

### 3.1 Tạo Vercel account
1. Đi tới [vercel.com](https://vercel.com)
2. Click "Sign up" → chọn "Continue with GitHub"
3. Authorize Vercel access

### 3.2 Deploy project
1. Click "Add New..." → "Project"
2. Import từ GitHub → chọn repository `DeepRPS`
3. Configure project:
   - **Framework Preset**: Other
   - **Root Directory**: `.` (default)
   - **Build Command**: `cd FE && npm install && npm run build`
   - **Output Directory**: `FE/dist`

### 3.3 Add Environment Variables
Trong Vercel deployment settings, thêm:
```
SUPABASE_URL=your_project_url_from_step_2.3
SUPABASE_ANON_KEY=your_anon_key_from_step_2.3
```

### 3.4 Deploy!
Click "Deploy" và đợi khoảng 2-3 phút.

## Bước 4: Test Deployment

1. Vercel sẽ cho bạn URL như: `https://deep-rps-xyz.vercel.app`
2. Mở URL và test game
3. Check console để xem API calls hoạt động
4. Chơi vài ván và refresh page - AI sẽ nhớ pattern của bạn!

## 🎉 DONE! 

**Tổng chi phí: $0 / FREE FOREVER**

- ✅ Frontend: Vercel free hosting
- ✅ Backend: Vercel serverless functions  
- ✅ Database: Supabase 500MB free PostgreSQL
- ✅ User data persistence: AI học và cải thiện

## Troubleshooting

### Nếu build failed:
```bash
# Local test
cd FE
npm install
npm run build
```

### Nếu API không hoạt động:
- Check Vercel Function logs
- Verify environment variables
- Check Supabase connection

### Update code:
```bash
git add .
git commit -m "Update features"
git push
```
Vercel sẽ tự động redeploy!

---

**🚀 Ready to play? Share your game URL với bạn bè!**