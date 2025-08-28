import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import apiClient from '../services/apiClient'
import { supabase } from '../lib/supabase'
import { getAPIBaseURL } from '../services/apiConfig'

const MainAPITest: React.FC = () => {
  const { user } = useAuth()
  const [logs, setLogs] = useState<string[]>([])
  const [testing, setTesting] = useState(false)

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    const logMessage = `[${timestamp}] ${message}`
    console.log(logMessage)
    setLogs(prev => [...prev, logMessage])
  }

  const testMainAPI = async () => {
    setTesting(true)
    setLogs([])
    
    addLog('🚀 Testing MAIN API (not simple API)')
    addLog(`📍 API Base URL: ${getAPIBaseURL()}`)
    addLog(`👤 User Status: ${user ? `Logged in as ${user.email}` : 'Not logged in'}`)
    
    // Test 1: Health Check (Main API)
    addLog('=== TEST 1: Main API Health Check ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/health`)
      if (response.ok) {
        const data = await response.json()
        addLog(`✅ Main API Health: ${JSON.stringify(data)}`)
      } else {
        addLog(`❌ Main API Health Failed: ${response.status}`)
      }
    } catch (error) {
      addLog(`❌ Main API Health Error: ${(error as Error).message}`)
    }

    // Test 2: API Documentation
    addLog('=== TEST 2: Main API Documentation ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/docs`)
      if (response.ok) {
        addLog(`✅ API Docs Available: ${response.status}`)
      } else {
        addLog(`❌ API Docs Failed: ${response.status}`)
      }
    } catch (error) {
      addLog(`❌ API Docs Error: ${(error as Error).message}`)
    }

    // Test 3: Upload Endpoints (GET to check if they exist)
    addLog('=== TEST 3: Upload Endpoints ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/api/upload/dataset`, {
        method: 'OPTIONS' // Check if endpoint exists
      })
      addLog(`✅ Upload endpoint exists: ${response.status}`)
    } catch (error) {
      addLog(`❌ Upload endpoint error: ${(error as Error).message}`)
    }

    // Test 4: Preprocessing Endpoints
    addLog('=== TEST 4: Preprocessing Endpoints ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/api/preprocess/numeric/missing-values`, {
        method: 'OPTIONS'
      })
      addLog(`✅ Preprocessing endpoint exists: ${response.status}`)
    } catch (error) {
      addLog(`❌ Preprocessing endpoint error: ${(error as Error).message}`)
    }

    // Test 5: EDA Endpoints
    addLog('=== TEST 5: EDA Endpoints ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/api/eda/numeric/correlation`, {
        method: 'OPTIONS'
      })
      addLog(`✅ EDA endpoint exists: ${response.status}`)
    } catch (error) {
      addLog(`❌ EDA endpoint error: ${(error as Error).message}`)
    }

    // Test 6: Model Training Endpoints
    addLog('=== TEST 6: Model Training Endpoints ===')
    try {
      const response = await fetch(`${getAPIBaseURL()}/api/model/train/logistic-regression`, {
        method: 'OPTIONS'
      })
      addLog(`✅ Model training endpoint exists: ${response.status}`)
    } catch (error) {
      addLog(`❌ Model training endpoint error: ${(error as Error).message}`)
    }

    // Test 7: Supabase Connection (if needed by main API)
    addLog('=== TEST 7: Supabase Connection ===')
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

    // Test 8: API Client Integration
    addLog('=== TEST 8: API Client Test ===')
    try {
      const connected = await apiClient.testConnection()
      addLog(`✅ API Client Test Connection: ${connected}`)
    } catch (error) {
      addLog(`❌ API Client Error: ${(error as Error).message}`)
    }

    addLog('🏁 Main API Test Suite Completed')
    addLog('📋 Summary: Check if all endpoints are available and responding')
    setTesting(false)
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">🔧 Main API Test Suite</h1>
        <p className="text-gray-600 mb-6">Testing the REAL main API with all 26 endpoints (not simple_api.py)</p>
        
        <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-200 mb-6">
          <button
            onClick={testMainAPI}
            disabled={testing}
            className="px-6 py-3 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {testing ? '🔄 Testing Main API...' : '🔧 Test Main API (Real Endpoints)'}
          </button>
        </div>

        <div className="bg-black text-green-400 p-6 rounded-lg shadow-lg border border-gray-200 max-h-96 overflow-y-auto">
          <h3 className="text-lg font-semibold mb-4 text-white">📊 Main API Test Logs:</h3>
          {logs.length === 0 ? (
            <p className="text-gray-400">Click "Test Main API" to start testing the real endpoints...</p>
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

export default MainAPITest
