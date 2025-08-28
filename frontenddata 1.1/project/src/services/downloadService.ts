/**
 * Download utility service for handling file downloads
 */

import { apiClient } from './apiClient'

export interface DownloadOptions {
  bucketName: string
  filePath: string
  userId: string
  filename?: string
}

/**
 * Download a file and trigger browser download
 */
export async function downloadFile(options: DownloadOptions): Promise<void> {
  try {
    // Get the file blob from API
    const blob = await apiClient.downloadFile(
      options.bucketName,
      options.filePath,
      options.userId
    )

    // Create download URL
    const url = window.URL.createObjectURL(blob)

    // Create temporary download link
    const link = document.createElement('a')
    link.href = url
    
    // Set filename - use provided filename or extract from filePath
    const filename = options.filename || options.filePath.split('/').pop() || 'download'
    link.download = filename

    // Trigger download
    document.body.appendChild(link)
    link.click()

    // Cleanup
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)

  } catch (error) {
    console.error('Download failed:', error)
    throw new Error(`Failed to download file: ${error}`)
  }
}

/**
 * Download a preprocessing result file
 */
export async function downloadPreprocessingResult(
  outputKey: string,
  userId: string,
  filename?: string
): Promise<void> {
  return downloadFile({
    bucketName: 'preprocessed',
    filePath: outputKey,
    userId,
    filename
  })
}

/**
 * Download an augmentation result file
 */
export async function downloadAugmentationResult(
  outputKey: string,
  userId: string,
  filename?: string
): Promise<void> {
  return downloadFile({
    bucketName: 'augmented',
    filePath: outputKey,
    userId,
    filename
  })
}
