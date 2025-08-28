import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import LandingPage from './components/LandingPage'
import DatasetSelectionPage from './components/DatasetSelectionPage'
import HomePage from './components/HomePage'
import PreprocessingPage from './components/PreprocessingPage'
import DataAugmentationPage from './components/DataAugmentationPage'
import ExploratoryDataAnalysisPage from './components/ExploratoryDataAnalysisPage'
import ModelTrainingPage from './components/ModelTrainingPage'
import ProtectedRoute from './components/ProtectedRoute'
import APITestComponent from './components/APITestComponent'
import ComprehensiveAPITest from './components/ComprehensiveAPITest'
import MainAPITest from './components/MainAPITest'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route 
            path="/dataset-selection" 
            element={
              <ProtectedRoute>
                <DatasetSelectionPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/home" 
            element={
              <ProtectedRoute>
                <HomePage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/preprocessing" 
            element={
              <ProtectedRoute>
                <PreprocessingPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/data-augmentation" 
            element={
              <ProtectedRoute>
                <DataAugmentationPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/exploratory-data-analysis" 
            element={
              <ProtectedRoute>
                <ExploratoryDataAnalysisPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/model-training" 
            element={
              <ProtectedRoute>
                <ModelTrainingPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/api-test" 
            element={
              <ProtectedRoute>
                <APITestComponent />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/comprehensive-test" 
            element={
              <ProtectedRoute>
                <ComprehensiveAPITest />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/main-api-test" 
            element={
              <ProtectedRoute>
                <MainAPITest />
              </ProtectedRoute>
            } 
          />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App