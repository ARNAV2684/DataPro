import { supabase, UserFile, AugmentedFile } from '../lib/supabase'

export interface FileUploadResult {
  file: UserFile
  uploadUrl?: string
}

export class FileService {
  /**
   * Upload a file to Supabase Storage and save metadata to database
   */
  static async uploadFile(
    file: File, 
    userId: string,
    processingStage: string = 'uploaded'
  ): Promise<FileUploadResult> {
    try {
      // Generate unique filename
      const timestamp = Date.now()
      const filename = `${userId}/${processingStage}/${timestamp}_${file.name}`
      
      // Upload to Supabase Storage
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('user-datasets')
        .upload(filename, file)
      
      if (uploadError) throw uploadError
      
      // Save metadata to database
      const { data: fileData, error: dbError } = await supabase
        .from('user_files')
        .insert({
          user_id: userId,
          filename: filename,
          original_name: file.name,
          file_type: file.type,
          storage_path: uploadData.path,
          file_size: file.size,
          processing_stage: processingStage
        })
        .select()
        .single()
      
      if (dbError) throw dbError
      
      // Get upload URL for immediate access if needed
      const { data: urlData } = supabase.storage
        .from('user-datasets')
        .getPublicUrl(uploadData.path)
      
      return {
        file: {
          id: fileData.id,
          userId: fileData.user_id,
          filename: fileData.filename,
          originalName: fileData.original_name,
          fileType: fileData.file_type,
          storagePath: fileData.storage_path,
          fileSize: fileData.file_size,
          processingStage: fileData.processing_stage,
          uploadDate: fileData.upload_date
        },
        uploadUrl: urlData.publicUrl
      }
    } catch (error) {
      console.error('File upload failed:', error)
      throw error
    }
  }

  /**
   * Get all files for a user, optionally filtered by processing stage
   */
  static async getUserFiles(
    userId: string, 
    processingStage?: string
  ): Promise<UserFile[]> {
    try {
      let query = supabase
        .from('user_files')
        .select('*')
        .eq('user_id', userId)
      
      if (processingStage) {
        query = query.eq('processing_stage', processingStage)
      }
      
      const { data, error } = await query.order('upload_date', { ascending: false })
      
      if (error) throw error
      
      return (data || []).map(item => ({
        id: item.id,
        userId: item.user_id,
        filename: item.filename,
        originalName: item.original_name,
        fileType: item.file_type,
        storagePath: item.storage_path,
        fileSize: item.file_size,
        processingStage: item.processing_stage,
        uploadDate: item.upload_date
      }))
    } catch (error) {
      console.error('Failed to get user files:', error)
      throw error
    }
  }

  /**
   * Download file content from Supabase Storage
   */
  static async downloadFileContent(storagePath: string): Promise<string> {
    try {
      const { data, error } = await supabase.storage
        .from('user-datasets')
        .download(storagePath)
      
      if (error) throw error
      
      // Convert blob to text
      const text = await data.text()
      return text
    } catch (error) {
      console.error('Failed to download file:', error)
      throw error
    }
  }

  /**
   * Save augmented results back to storage and database
   */
  static async saveAugmentedResults(
    originalFileId: string,
    technique: string,
    parameters: Record<string, any>,
    augmentedSamples: any[]
  ): Promise<AugmentedFile> {
    try {
      // Convert augmented samples to CSV
      const csvContent = this.convertSamplesToCSV(augmentedSamples)
      
      // Create blob and file
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const fileName = `augmented_${technique}_${Date.now()}.csv`
      
      // Get original file info to determine user
      const { data: originalFile } = await supabase
        .from('user_files')
        .select('user_id')
        .eq('id', originalFileId)
        .single()
      
      if (!originalFile) throw new Error('Original file not found')
      
      // Upload augmented file
      const file = new File([blob], fileName, { type: 'text/csv' })
      const uploadResult = await this.uploadFile(file, originalFile.user_id, 'augmented')
      
      // Save augmentation metadata
      const { data: augmentedFile, error } = await supabase
        .from('augmented_files')
        .insert({
          original_file_id: originalFileId,
          technique: technique,
          parameters: parameters,
          result_storage_path: uploadResult.file.storagePath,
          sample_count: augmentedSamples.length
        })
        .select()
        .single()
      
      if (error) throw error
      
      return {
        id: augmentedFile.id,
        originalFileId: augmentedFile.original_file_id,
        technique: augmentedFile.technique,
        parameters: augmentedFile.parameters,
        resultStoragePath: augmentedFile.result_storage_path,
        sampleCount: augmentedFile.sample_count,
        createdAt: augmentedFile.created_at
      }
    } catch (error) {
      console.error('Failed to save augmented results:', error)
      throw error
    }
  }

  /**
   * Convert augmented samples to CSV format
   */
  private static convertSamplesToCSV(samples: any[]): string {
    if (samples.length === 0) return ''
    
    // Get headers from first sample
    const headers = Object.keys(samples[0])
    const csvHeaders = headers.join(',')
    
    // Convert each sample to CSV row
    const csvRows = samples.map(sample => 
      headers.map(header => {
        const value = sample[header]
        // Escape commas and quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value
      }).join(',')
    )
    
    return [csvHeaders, ...csvRows].join('\n')
  }

  /**
   * Parse CSV content into array of objects
   */
  static parseCSV(csvContent: string): Record<string, any>[] {
    const lines = csvContent.trim().split('\n')
    if (lines.length < 2) return []
    
    const headers = lines[0].split(',').map(h => h.trim())
    const data = []
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim())
      const row: Record<string, any> = {}
      
      headers.forEach((header, index) => {
        row[header] = values[index] || ''
      })
      
      data.push(row)
    }
    
    return data
  }

  /**
   * Extract text content from parsed CSV data
   */
  static extractTextFromCSV(parsedData: Record<string, any>[]): string[] {
    const textColumns = Object.keys(parsedData[0] || {}).filter(key => 
      // Identify text columns (non-numeric)
      parsedData.some(row => isNaN(Number(row[key])) && row[key].length > 0)
    )
    
    const texts: string[] = []
    
    parsedData.forEach(row => {
      textColumns.forEach(column => {
        if (row[column] && typeof row[column] === 'string' && row[column].trim().length > 0) {
          texts.push(row[column].trim())
        }
      })
    })
    
    return texts
  }
}
