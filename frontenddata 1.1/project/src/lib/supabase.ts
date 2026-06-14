import { createClient } from '@supabase/supabase-js'

// Supabase browser client. Credentials come from Vite env vars at build time:
//   VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
// Placeholders keep the app from crashing on import when they are unset; auth /
// storage calls will simply fail until real values are provided.
const supabaseUrl = (import.meta.env.VITE_SUPABASE_URL as string) || 'https://placeholder.supabase.co'
const supabaseAnonKey = (import.meta.env.VITE_SUPABASE_ANON_KEY as string) || 'public-anon-key'

if (!import.meta.env.VITE_SUPABASE_URL || !import.meta.env.VITE_SUPABASE_ANON_KEY) {
  console.warn(
    '[supabase] VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY are not set. ' +
    'Auth and storage will not work until they are configured.'
  )
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// ---------------------------------------------------------------------------
// Shared row types (camelCase, as mapped by FileService)
// ---------------------------------------------------------------------------
export interface UserFile {
  id: string
  userId: string
  filename: string
  originalName: string
  fileType: string
  storagePath: string
  fileSize: number
  processingStage: string
  uploadDate: string
}

export interface AugmentedFile {
  id: string
  originalFileId: string
  technique: string
  parameters: Record<string, any>
  resultStoragePath: string
  sampleCount: number
  createdAt: string
}
