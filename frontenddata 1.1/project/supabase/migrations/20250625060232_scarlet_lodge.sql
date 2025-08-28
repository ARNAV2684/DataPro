/*
  # Complete User Session & Progress Tracking System

  1. New Tables
    - `user_preferences` (enhanced)
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `dataset_type` (text, the selected data type)
      - `theme` (text, UI theme preference)
      - `notifications_enabled` (boolean)
      - `auto_save` (boolean)
      - `language` (text, preferred language)
      - `created_at` (timestamp)
      - `updated_at` (timestamp)
    
    - `user_sessions`
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `session_token` (text, unique session identifier)
      - `login_time` (timestamp)
      - `last_activity` (timestamp)
      - `ip_address` (text)
      - `device_info` (jsonb, browser/device details)
      - `is_active` (boolean)
      - `expires_at` (timestamp)
      - `logout_time` (timestamp, nullable)
      - `logout_reason` (text, nullable)
    
    - `user_progress`
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `activity_type` (text, type of activity)
      - `activity_data` (jsonb, flexible activity data)
      - `progress_state` (jsonb, current progress status)
      - `page_visited` (text, nullable)
      - `model_trained` (text, nullable)
      - `completion_percentage` (integer)
      - `is_completed` (boolean)
      - `created_at` (timestamp)
    
    - `user_profiles`
      - `id` (uuid, primary key)
      - `user_id` (uuid, foreign key to auth.users)
      - `first_name` (text, nullable)
      - `last_name` (text, nullable)
      - `display_name` (text, nullable)
      - `bio` (text, nullable)
      - `avatar_url` (text, nullable)
      - `onboarding_completed` (boolean, default false)
      - `first_login` (boolean, default true)
      - `total_sessions` (integer, default 0)
      - `total_models_trained` (integer, default 0)
      - `last_login` (timestamp, nullable)
      - `created_at` (timestamp)
      - `updated_at` (timestamp)

  2. Security
    - Enable RLS on all tables
    - Add policies for users to read/write their own data
    - Add session validation functions
*/

-- =====================================================
-- 1. ENHANCED USER PREFERENCES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS user_preferences (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  dataset_type text NOT NULL CHECK (dataset_type IN ('numeric', 'text', 'image', 'mixed')),
  theme text DEFAULT 'light' CHECK (theme IN ('light', 'dark', 'auto')),
  notifications_enabled boolean DEFAULT true,
  auto_save boolean DEFAULT true,
  language text DEFAULT 'en' CHECK (language IN ('en', 'es', 'fr', 'de', 'zh')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- =====================================================
-- 2. USER SESSIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS user_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  session_token text UNIQUE NOT NULL,
  login_time timestamptz DEFAULT now(),
  last_activity timestamptz DEFAULT now(),
  ip_address inet,
  device_info jsonb DEFAULT '{}',
  is_active boolean DEFAULT true,
  expires_at timestamptz DEFAULT (now() + interval '7 days'),
  logout_time timestamptz,
  logout_reason text CHECK (logout_reason IN ('user_logout', 'session_expired', 'forced_logout', 'inactivity'))
);

-- =====================================================
-- 3. USER PROGRESS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS user_progress (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  session_id uuid REFERENCES user_sessions(id) ON DELETE SET NULL,
  activity_type text NOT NULL CHECK (activity_type IN (
    'login', 'logout', 'page_visit', 'model_training', 'data_upload', 
    'tutorial_start', 'tutorial_complete', 'onboarding_step', 
    'feature_used', 'error_encountered', 'file_download'
  )),
  activity_data jsonb DEFAULT '{}',
  progress_state jsonb DEFAULT '{}',
  page_visited text,
  model_trained text,
  completion_percentage integer DEFAULT 0 CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
  is_completed boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- =====================================================
-- 4. USER PROFILES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS user_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
  first_name text,
  last_name text,
  display_name text,
  bio text,
  avatar_url text,
  onboarding_completed boolean DEFAULT false,
  first_login boolean DEFAULT true,
  total_sessions integer DEFAULT 0,
  total_models_trained integer DEFAULT 0,
  last_login timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- =====================================================
-- 5. ROW LEVEL SECURITY (RLS) SETUP
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 6. RLS POLICIES FOR USER_PREFERENCES
-- =====================================================

-- Policy for users to read their own preferences
CREATE POLICY "Users can read own preferences"
  ON user_preferences
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

-- Policy for users to insert their own preferences
CREATE POLICY "Users can insert own preferences"
  ON user_preferences
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Policy for users to update their own preferences
CREATE POLICY "Users can update own preferences"
  ON user_preferences
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =====================================================
-- 7. RLS POLICIES FOR USER_SESSIONS
-- =====================================================

-- Policy for users to read their own sessions
CREATE POLICY "Users can read own sessions"
  ON user_sessions
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

-- Policy for users to insert their own sessions
CREATE POLICY "Users can insert own sessions"
  ON user_sessions
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Policy for users to update their own sessions
CREATE POLICY "Users can update own sessions"
  ON user_sessions
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =====================================================
-- 8. RLS POLICIES FOR USER_PROGRESS
-- =====================================================

-- Policy for users to read their own progress
CREATE POLICY "Users can read own progress"
  ON user_progress
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

-- Policy for users to insert their own progress
CREATE POLICY "Users can insert own progress"
  ON user_progress
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Policy for users to update their own progress
CREATE POLICY "Users can update own progress"
  ON user_progress
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =====================================================
-- 9. RLS POLICIES FOR USER_PROFILES
-- =====================================================

-- Policy for users to read their own profiles
CREATE POLICY "Users can read own profiles"
  ON user_profiles
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

-- Policy for users to insert their own profiles
CREATE POLICY "Users can insert own profiles"
  ON user_profiles
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Policy for users to update their own profiles
CREATE POLICY "Users can update own profiles"
  ON user_profiles
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =====================================================
-- 10. HELPER FUNCTIONS
-- =====================================================

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION create_user_profile()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO user_profiles (user_id, display_name, created_at)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data->>'full_name', 'User'), now());
  
  INSERT INTO user_preferences (user_id, dataset_type, created_at)
  VALUES (NEW.id, 'text', now());
  
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to validate session token
CREATE OR REPLACE FUNCTION is_session_valid(session_token_input text)
RETURNS boolean AS $$
DECLARE
  session_exists boolean;
BEGIN
  SELECT EXISTS(
    SELECT 1 FROM user_sessions 
    WHERE session_token = session_token_input 
    AND is_active = true 
    AND expires_at > now()
  ) INTO session_exists;
  
  RETURN session_exists;
END;
$$ language 'plpgsql';

-- Function to update session activity
CREATE OR REPLACE FUNCTION update_session_activity(session_token_input text)
RETURNS void AS $$
BEGIN
  UPDATE user_sessions 
  SET last_activity = now()
  WHERE session_token = session_token_input 
  AND is_active = true;
END;
$$ language 'plpgsql';

-- Function to expire old sessions
CREATE OR REPLACE FUNCTION expire_old_sessions()
RETURNS void AS $$
BEGIN
  UPDATE user_sessions 
  SET is_active = false, 
      logout_time = now(), 
      logout_reason = 'session_expired'
  WHERE expires_at < now() 
  AND is_active = true;
END;
$$ language 'plpgsql';

-- =====================================================
-- 11. TRIGGERS
-- =====================================================

-- Trigger to automatically update updated_at for user_preferences
CREATE TRIGGER update_user_preferences_updated_at
  BEFORE UPDATE ON user_preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Trigger to automatically update updated_at for user_profiles
CREATE TRIGGER update_user_profiles_updated_at
  BEFORE UPDATE ON user_profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Trigger to create user profile on signup
CREATE TRIGGER create_user_profile_trigger
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION create_user_profile();

-- =====================================================
-- 12. INDEXES FOR PERFORMANCE
-- =====================================================

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at);

CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_activity_type ON user_progress(activity_type);
CREATE INDEX IF NOT EXISTS idx_user_progress_created_at ON user_progress(created_at);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);

-- =====================================================
-- 13. SAMPLE DATA INSERTION FUNCTIONS (OPTIONAL)
-- =====================================================

-- Function to log user activity
CREATE OR REPLACE FUNCTION log_user_activity(
  p_user_id uuid,
  p_session_id uuid,
  p_activity_type text,
  p_activity_data jsonb DEFAULT '{}',
  p_page_visited text DEFAULT NULL,
  p_model_trained text DEFAULT NULL
)
RETURNS uuid AS $$
DECLARE
  progress_id uuid;
BEGIN
  INSERT INTO user_progress (
    user_id, session_id, activity_type, activity_data, 
    page_visited, model_trained, created_at
  )
  VALUES (
    p_user_id, p_session_id, p_activity_type, p_activity_data,
    p_page_visited, p_model_trained, now()
  )
  RETURNING id INTO progress_id;
  
  RETURN progress_id;
END;
$$ language 'plpgsql';