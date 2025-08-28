console.log('📡 Testing API Client directly...')

// Import our API client
import apiClient from './src/services/apiClient.js'
import { getAPIBaseURL } from './src/services/apiConfig.js'

async function testAPIClient() {
  console.log('🔧 API Configuration:')
  console.log('Base URL:', getAPIBaseURL())
  
  console.log('\n📡 Testing API Client methods...')
  
  try {
    // Test connection
    console.log('Testing testConnection()...')
    const result = await apiClient.testConnection()
    console.log('✅ API Connection successful:', result)
  } catch (error) {
    console.log('❌ API Connection failed:', error.message)
  }
  
  try {
    // Test health endpoint
    console.log('\nTesting manual fetch to health endpoint...')
    const response = await fetch(getAPIBaseURL() + '/health')
    const data = await response.json()
    console.log('✅ Health check successful:', data)
  } catch (error) {
    console.log('❌ Health check failed:', error.message)
  }
}

testAPIClient()
