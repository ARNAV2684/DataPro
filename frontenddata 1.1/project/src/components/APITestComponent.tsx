import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import apiClient from '../services/apiClient'
import { supabase } from '../lib/supabase'
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react'

interface ConnectionStatus {
  api: boolean
  supabase: boolean
  auth: boolean
}

const APITestComponent: React.FC = () => {
  const { user } = useAuth()
  const [testing, setTesting] = useState(false)
  const [status, setStatus] = useState<ConnectionStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const testConnections = async () => {
    setTesting(true)
    setError(null)
    console.log('🧪 Starting connection tests...')
    
    try {
      const newStatus: ConnectionStatus = {
        api: false,
        supabase: false,
        auth: false
      }

      // Test API connection
      console.log('📡 Testing API connection...')
      try {
        await apiClient.testConnection()
        console.log('✅ API connection successful')
        newStatus.api = true
      } catch (e) {
        console.error('❌ API test failed:', e)
        console.log('ℹ️ This is expected if backend is not running')
      }

      // Test Supabase connection (basic)
      console.log('🗄️ Testing Supabase connection...')
      try {
        // Test actual Supabase connection
        const { data, error } = await supabase.from('datasets').select('id').limit(1)
        if (error) {
          console.error('❌ Supabase error:', error)
        } else {
          console.log('✅ Supabase connection successful:', data)
          newStatus.supabase = true
        }
      } catch (e) {
        console.error('❌ Supabase test failed:', e)
      }

      // Test authentication
      console.log('🔐 Testing authentication...')
      newStatus.auth = !!user
      if (user) {
        console.log('✅ User authenticated:', { id: user.id, email: user.email })
      } else {
        console.log('⚠️ No user authenticated')
      }

      setStatus(newStatus)
      console.log('📊 Final status:', newStatus)
    } catch (e) {
      console.error('❌ Overall test error:', e)
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setTesting(false)
      console.log('🏁 Connection tests completed')
    }
  }

  // Auto-test on component mount
  useEffect(() => {
    testConnections()
  }, [user])

  const StatusIcon: React.FC<{ status: boolean }> = ({ status }) => {
    if (status) {
      return <CheckCircle2 className="w-5 h-5 text-green-500" />
    }
    return <XCircle className="w-5 h-5 text-red-500" />
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
        <button
          onClick={testConnections}
          disabled={testing}
          className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {testing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Testing...
            </>
          ) : (
            'Test Connections'
          )}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {status && (
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
            <span className="text-gray-700">ML Pipeline API</span>
            <div className="flex items-center gap-2">
              <StatusIcon status={status.api} />
              <span className="text-sm text-gray-600">
                {status.api ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
            <span className="text-gray-700">Supabase Database</span>
            <div className="flex items-center gap-2">
              <StatusIcon status={status.supabase} />
              <span className="text-sm text-gray-600">
                {status.supabase ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
            <span className="text-gray-700">User Authentication</span>
            <div className="flex items-center gap-2">
              <StatusIcon status={status.auth} />
              <span className="text-sm text-gray-600">
                {status.auth ? `Signed in as ${user?.email}` : 'Not signed in'}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
        <p className="text-blue-700 text-sm">
          <strong>Phase 3 Batch 1: API Connection Status</strong><br />
          This component tests the connection to all system components.
        </p>
      </div>
    </div>
  )
}

export default APITestComponent
