-- =====================================================
-- GARUDA ML PIPELINE DATABASE MIGRATION
-- =====================================================
-- This migration creates the complete database structure for the Garuda ML Pipeline
-- Run this in your Supabase SQL Editor

-- =====================================================
-- EXTENSION SETUP
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- ENUMS
-- =====================================================

-- Pipeline stages
CREATE TYPE pipeline_stage AS ENUM (
    'upload',
    'preprocess', 
    'augment',
    'eda',
    'model'
);

-- Data types
CREATE TYPE data_type AS ENUM (
    'numeric',
    'text',
    'image',
    'mixed'
);

-- Processing status
CREATE TYPE processing_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL UNIQUE, -- References auth.users.id as text
    email TEXT,
    full_name TEXT,
    avatar_url TEXT,
    subscription_tier TEXT DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Datasets table - stores dataset metadata
CREATE TABLE IF NOT EXISTS public.datasets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL, -- References user_profiles.user_id
    name TEXT NOT NULL,
    description TEXT,
    original_filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    data_type data_type NOT NULL,
    file_size BIGINT NOT NULL,
    bucket_key TEXT NOT NULL, -- Storage path in Supabase bucket
    status processing_status DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pipeline artifacts table - stores all pipeline outputs
CREATE TABLE IF NOT EXISTS public.pipeline_artifacts (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL, -- References user_profiles.user_id
    dataset_id TEXT NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
    stage pipeline_stage NOT NULL,
    operation TEXT NOT NULL, -- e.g., 'handle_missing_values', 'smote', etc.
    bucket_key TEXT NOT NULL, -- Storage path in Supabase bucket
    status processing_status DEFAULT 'pending',
    input_artifact_id TEXT REFERENCES public.pipeline_artifacts(id), -- Links to previous stage
    parameters JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    execution_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data samples table - stores actual data for quick preview/analysis
CREATE TABLE IF NOT EXISTS public.data_samples (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    artifact_id TEXT REFERENCES public.pipeline_artifacts(id) ON DELETE CASCADE,
    sample_data JSONB NOT NULL, -- Stores sample rows as JSON
    total_rows INTEGER NOT NULL,
    total_columns INTEGER NOT NULL,
    column_info JSONB DEFAULT '{}', -- Column names, types, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Processing logs table - stores detailed execution logs
CREATE TABLE IF NOT EXISTS public.processing_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    artifact_id TEXT REFERENCES public.pipeline_artifacts(id) ON DELETE CASCADE,
    log_level TEXT DEFAULT 'info', -- info, warning, error
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS public.data_quality_metrics (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    artifact_id TEXT REFERENCES public.pipeline_artifacts(id) ON DELETE CASCADE,
    missing_values JSONB DEFAULT '{}', -- Per column missing value counts
    outliers JSONB DEFAULT '{}', -- Outlier detection results
    duplicates INTEGER DEFAULT 0,
    data_types JSONB DEFAULT '{}', -- Column data types
    statistics JSONB DEFAULT '{}', -- Basic statistics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model training results table
CREATE TABLE IF NOT EXISTS public.model_results (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL, -- References user_profiles.user_id
    dataset_id TEXT REFERENCES public.datasets(id) ON DELETE CASCADE,
    model_type TEXT NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}', -- accuracy, precision, recall, etc.
    model_path TEXT, -- Storage path for saved model
    status processing_status DEFAULT 'pending',
    training_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Datasets indexes
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON public.datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_data_type ON public.datasets(data_type);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON public.datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON public.datasets(created_at);

-- Pipeline artifacts indexes
CREATE INDEX IF NOT EXISTS idx_artifacts_user_id ON public.pipeline_artifacts(user_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_dataset_id ON public.pipeline_artifacts(dataset_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_stage ON public.pipeline_artifacts(stage);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON public.pipeline_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_artifacts_input_id ON public.pipeline_artifacts(input_artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON public.pipeline_artifacts(created_at);

-- Data samples indexes
CREATE INDEX IF NOT EXISTS idx_samples_artifact_id ON public.data_samples(artifact_id);

-- Processing logs indexes
CREATE INDEX IF NOT EXISTS idx_logs_artifact_id ON public.processing_logs(artifact_id);
CREATE INDEX IF NOT EXISTS idx_logs_level ON public.processing_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON public.processing_logs(created_at);

-- Model results indexes
CREATE INDEX IF NOT EXISTS idx_models_user_id ON public.model_results(user_id);
CREATE INDEX IF NOT EXISTS idx_models_dataset_id ON public.model_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_models_status ON public.model_results(status);

-- =====================================================
-- ROW LEVEL SECURITY (RLS)
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pipeline_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.data_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.data_quality_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_results ENABLE ROW LEVEL SECURITY;

-- User profiles policies
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own profile" ON public.user_profiles
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Datasets policies
CREATE POLICY "Users can view own datasets" ON public.datasets
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own datasets" ON public.datasets
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own datasets" ON public.datasets
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can delete own datasets" ON public.datasets
    FOR DELETE USING (auth.uid()::text = user_id);

-- Pipeline artifacts policies
CREATE POLICY "Users can view own artifacts" ON public.pipeline_artifacts
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own artifacts" ON public.pipeline_artifacts
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own artifacts" ON public.pipeline_artifacts
    FOR UPDATE USING (auth.uid()::text = user_id);

-- Data samples policies
CREATE POLICY "Users can view own data samples" ON public.data_samples
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = data_samples.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert own data samples" ON public.data_samples
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = data_samples.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

-- Processing logs policies  
CREATE POLICY "Users can view own processing logs" ON public.processing_logs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = processing_logs.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert own processing logs" ON public.processing_logs
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = processing_logs.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

-- Data quality metrics policies
CREATE POLICY "Users can view own quality metrics" ON public.data_quality_metrics
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = data_quality_metrics.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert own quality metrics" ON public.data_quality_metrics
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.pipeline_artifacts 
            WHERE id = data_quality_metrics.artifact_id 
            AND user_id = auth.uid()::text
        )
    );

-- Model results policies
CREATE POLICY "Users can view own model results" ON public.model_results
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own model results" ON public.model_results
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own model results" ON public.model_results
    FOR UPDATE USING (auth.uid()::text = user_id);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON public.user_profiles 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at 
    BEFORE UPDATE ON public.datasets 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_artifacts_updated_at 
    BEFORE UPDATE ON public.pipeline_artifacts 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_models_updated_at 
    BEFORE UPDATE ON public.model_results 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- =====================================================
-- STORAGE BUCKET POLICIES
-- =====================================================

-- Create storage buckets (run these in Supabase Dashboard > Storage)
-- 1. datasets - for uploaded raw data
-- 2. preprocessed - for processed data  
-- 3. augmented - for augmented data
-- 4. models - for saved models
-- 5. visualizations - for charts and plots

-- Example storage policies (adapt for each bucket):
-- 
-- Policy name: "Users can manage own files"
-- SELECT: true, INSERT: true, UPDATE: true, DELETE: true
-- Target roles: authenticated
-- Policy definition: (bucket_id = 'datasets'::text AND (storage.foldername(name))[1] = auth.uid()::text)

-- =====================================================
-- UTILITY VIEWS
-- =====================================================

-- Simple view for pipeline status (create after tables exist)
-- Note: Create this view manually after running the main migration if needed

-- View for complete pipeline status
-- CREATE OR REPLACE VIEW public.pipeline_overview AS
-- SELECT 
--     d.id as dataset_id,
--     d.user_id,
--     d."name" as dataset_name,
--     d.data_type,
--     d.file_size,
--     d.created_at as uploaded_at,
--     COUNT(pa.id) as total_operations,
--     COUNT(CASE WHEN pa.status = 'completed' THEN 1 END) as completed_operations,
--     COUNT(CASE WHEN pa.status = 'failed' THEN 1 END) as failed_operations,
--     MAX(pa.updated_at) as last_operation
-- FROM public.datasets AS d
-- LEFT JOIN public.pipeline_artifacts AS pa ON d.id = pa.dataset_id
-- GROUP BY d.id, d.user_id, d."name", d.data_type, d.file_size, d.created_at;

-- View for latest artifacts per stage
-- CREATE OR REPLACE VIEW public.latest_artifacts AS
-- SELECT DISTINCT ON (dataset_id, stage) 
--     dataset_id,
--     stage,
--     operation,
--     bucket_key,
--     status,
--     created_at
-- FROM public.pipeline_artifacts
-- ORDER BY dataset_id, stage, created_at DESC;

-- =====================================================
-- SAMPLE DATA AND TESTING
-- =====================================================

-- Insert sample data types for testing (optional)
-- INSERT INTO public.user_profiles (id, email, full_name) 
-- VALUES (auth.uid(), 'test@example.com', 'Test User')
-- ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- MIGRATION COMPLETE
-- =====================================================
-- 
-- After running this migration:
-- 1. Create the storage buckets in Supabase Dashboard
-- 2. Set up storage policies for each bucket
-- 3. Test the database structure
-- 4. Update your application code to use these tables
