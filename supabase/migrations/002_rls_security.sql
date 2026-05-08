-- Enable Row Level Security (RLS) on all core tables
ALTER TABLE IF EXISTS user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS messages ENABLE ROW LEVEL SECURITY;

-- 1. User Profiles RLS
-- Users can only read their own profile
CREATE POLICY "Users can read own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = id);

-- Users can only update their own profile
CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- Users can only insert their own profile
CREATE POLICY "Users can insert own profile" ON user_profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- 2. Sessions RLS
-- Users can only read their own sessions
CREATE POLICY "Users can read own sessions" ON sessions
    FOR SELECT USING (auth.uid() = user_id);

-- Users can only insert their own sessions
CREATE POLICY "Users can insert own sessions" ON sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can only update their own sessions
CREATE POLICY "Users can update own sessions" ON sessions
    FOR UPDATE USING (auth.uid() = user_id);

-- 3. Messages RLS
-- Users can only read their own messages
CREATE POLICY "Users can read own messages" ON messages
    FOR SELECT USING (
        session_id IN (
            SELECT id FROM sessions WHERE user_id = auth.uid()
        )
    );

-- Users can only insert their own messages
CREATE POLICY "Users can insert own messages" ON messages
    FOR INSERT WITH CHECK (
        session_id IN (
            SELECT id FROM sessions WHERE user_id = auth.uid()
        )
    );

-- Prevent any DELETE operations
-- Mental health records should be retained unless explicitly requested for complete account deletion
CREATE POLICY "Prevent delete on user profiles" ON user_profiles FOR DELETE USING (false);
CREATE POLICY "Prevent delete on sessions" ON sessions FOR DELETE USING (false);
CREATE POLICY "Prevent delete on messages" ON messages FOR DELETE USING (false);
