// Test file to verify Supabase integration
import { supabase } from './src/lib/supabase'

async function testSupabaseConnection() {
  console.log('🧪 Testing Supabase connection...')
  
  try {
    // Test 1: Check if we can connect to Supabase
    const { data: { session } } = await supabase.auth.getSession()
    console.log('✅ Supabase connection successful')
    console.log('📱 Session status:', session ? 'Authenticated' : 'Not authenticated')
    
    // Test 2: Check if we can query tables (this will only work after running the SQL migration)
    try {
      const { data: files, error } = await supabase
        .from('user_files')
        .select('*')
        .limit(1)
      
      if (error) {
        console.log('⚠️  Tables not yet created:', error.message)
        console.log('📋 Please run the SQL migration in Supabase Dashboard')
      } else {
        console.log('✅ Database tables accessible')
        console.log('📊 Sample query result:', files)
      }
    } catch (tableError) {
      console.log('⚠️  Database schema needs setup:', tableError)
    }
    
    // Test 3: Check storage
    try {
      const { data: buckets, error: bucketError } = await supabase.storage.listBuckets()
      if (bucketError) {
        console.log('⚠️  Storage access error:', bucketError.message)
      } else {
        console.log('✅ Storage accessible')
        const userDatasetsBucket = buckets.find(b => b.name === 'user-datasets')
        if (userDatasetsBucket) {
          console.log('✅ user-datasets bucket found')
        } else {
          console.log('⚠️  user-datasets bucket not found - create it in Supabase Dashboard')
        }
      }
    } catch (storageError) {
      console.log('⚠️  Storage test error:', storageError)
    }
    
  } catch (error) {
    console.error('❌ Supabase connection failed:', error)
  }
}

// Run the test if this file is executed directly
if (typeof window === 'undefined') {
  testSupabaseConnection()
}

export { testSupabaseConnection }
