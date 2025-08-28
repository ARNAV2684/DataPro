-- Create storage bucket for user datasets (run this first)
-- Go to Supabase Dashboard > Storage > Create new bucket
-- Bucket name: user-datasets
-- Public: false (for security)

-- Create the user_files table
CREATE TABLE IF NOT EXISTS public.user_files (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    original_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    processing_stage TEXT NOT NULL DEFAULT 'uploaded',
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create the augmented_files table
CREATE TABLE IF NOT EXISTS public.augmented_files (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    original_file_id TEXT REFERENCES public.user_files(id) ON DELETE CASCADE,
    technique TEXT NOT NULL,
    parameters JSONB NOT NULL,
    result_storage_path TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create the processing_jobs table
CREATE TABLE IF NOT EXISTS public.processing_jobs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    parameters JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON public.user_files(user_id);
CREATE INDEX IF NOT EXISTS idx_user_files_processing_stage ON public.user_files(processing_stage);
CREATE INDEX IF NOT EXISTS idx_augmented_files_original_file_id ON public.augmented_files(original_file_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_user_id ON public.processing_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON public.processing_jobs(status);

-- Enable Row Level Security (RLS)
ALTER TABLE public.user_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.augmented_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.processing_jobs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for user_files
CREATE POLICY "Users can view own files" ON public.user_files
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own files" ON public.user_files
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own files" ON public.user_files
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own files" ON public.user_files
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for augmented_files
CREATE POLICY "Users can view own augmented files" ON public.augmented_files
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.user_files 
            WHERE id = augmented_files.original_file_id 
            AND user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert own augmented files" ON public.augmented_files
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.user_files 
            WHERE id = augmented_files.original_file_id 
            AND user_id = auth.uid()
        )
    );

-- Create RLS policies for processing_jobs
CREATE POLICY "Users can view own jobs" ON public.processing_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own jobs" ON public.processing_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own jobs" ON public.processing_jobs
    FOR UPDATE USING (auth.uid() = user_id);

-- Create storage policies (run these in the Storage > Policies section)
-- Policy for user-datasets bucket:
-- 
-- Policy name: "Users can upload own files"
-- SELECT: true
-- INSERT: true
-- UPDATE: true
-- DELETE: true
-- Target roles: authenticated
-- Policy definition: (bucket_id = 'user-datasets'::text AND (storage.foldername(name))[1] = auth.uid()::text)

-- Update function for updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER handle_processing_jobs_updated_at 
    BEFORE UPDATE ON public.processing_jobs 
    FOR EACH ROW EXECUTE PROCEDURE public.handle_updated_at();
