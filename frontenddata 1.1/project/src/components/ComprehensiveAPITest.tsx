import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import apiClient from '../services/apiClient'
import { supabase } from '../lib/supabase'
import { getAPIBaseURL } from '../services/apiConfig'

const ComprehensiveAPITest: React.FC = () => {
  const { user } = useAuth()
  const [logs, setLogs] = useState<string[]>([])
  const [testing, setTesting] = useState(false)

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    const logMessage = `[${timestamp}] ${message}`
    console.log(logMessage)
    setLogs(prev => [...prev, logMessage])
  }

  const runComprehensiveTest = async () => {
    setTesting(true)
    setLogs([])
    
    addLog('🚀 Starting Comprehensive API Test Suite')
    addLog(`📍 API Base URL: ${getAPIBaseURL()}`)
    addLog(`👤 User Status: ${user ? `Logged in as ${user.email}` : 'Not logged in'}`)
    
    // Test 1: Basic Fetch to Health Endpoint
    addLog('=== TEST 1: Direct Fetch to Health Endpoint ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/health`)
      addLog(`✅ Response Status: ${response.status} ${response.statusText}`)
      
      if (response.ok) {
        const data = await response.json()
        addLog(`✅ Response Data: ${JSON.stringify(data)}`)
      } else {
        addLog(`❌ Response not OK: ${response.status}`)
      }
    } catch (error) {
      addLog(`❌ Direct Fetch Failed: ${(error as Error).message}`)
    }

    // Test 2: API Client Health Check
    addLog('=== TEST 2: API Client Health Check ===')
    try {
      const result = await apiClient.healthCheck()
      addLog(`✅ API Client Health Check: ${JSON.stringify(result)}`)
    } catch (error) {
      addLog(`❌ API Client Health Check Failed: ${(error as Error).message}`)
    }

    // Test 3: API Client Test Connection
    addLog('=== TEST 3: API Client Test Connection ===')
    try {
      const connected = await apiClient.testConnection()
      addLog(`✅ Test Connection Result: ${connected}`)
    } catch (error) {
      addLog(`❌ Test Connection Failed: ${(error as Error).message}`)
    }

    // Test 4: API Test Endpoint
    addLog('=== TEST 4: API Test Endpoint ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/api/test`)
      if (response.ok) {
        const data = await response.json()
        addLog(`✅ API Test Endpoint: ${JSON.stringify(data)}`)
      } else {
        addLog(`❌ API Test Endpoint Failed: ${response.status}`)
      }
    } catch (error) {
      addLog(`❌ API Test Endpoint Error: ${(error as Error).message}`)
    }

    // Test 5: CORS Test
    addLog('=== TEST 5: CORS Test ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Origin': window.location.origin
        }
      })
      if (response.ok) {
        const data = await response.json()
        addLog(`✅ CORS Test: ${JSON.stringify(data)}`)
      } else {
        addLog(`❌ CORS Test Failed: ${response.status}`)
      }
    } catch (error) {
      addLog(`❌ CORS Test Error: ${(error as Error).message}`)
    }

    // Test 6: Supabase Connection
    addLog('=== TEST 6: Supabase Connection ===')
    try {
      const { data, error } = await supabase.from('datasets').select('id').limit(1)
      if (error) {
        addLog(`❌ Supabase Error: ${error.message}`)
      } else {
        addLog(`✅ Supabase Connected: Found ${data?.length || 0} records`)
      }
    } catch (error) {
      addLog(`❌ Supabase Connection Error: ${(error as Error).message}`)
    }

    addLog('🏁 Comprehensive Test Suite Completed')
    setTesting(false)
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">🧪 Comprehensive API Test Suite</h1>
        
        <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-200 mb-6">
          <button
            onClick={runComprehensiveTest}
            disabled={testing}
            className="px-6 py-3 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {testing ? '🔄 Testing...' : '🚀 Run Comprehensive Test'}
          </button>
        </div>

        <div className="bg-black text-green-400 p-6 rounded-lg shadow-lg border border-gray-200 max-h-96 overflow-y-auto">
          <h3 className="text-lg font-semibold mb-4 text-white">📊 Test Logs:</h3>
          {logs.length === 0 ? (
            <p className="text-gray-400">Click "Run Comprehensive Test" to start testing...</p>
          ) : (
            <div className="space-y-1 font-mono text-sm">
              {logs.map((log, index) => (
                <div key={index} className="whitespace-pre-wrap">
                  {log}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ComprehensiveAPITest
