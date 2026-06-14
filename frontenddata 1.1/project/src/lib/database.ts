// Lightweight persistence for small UI preferences.
//
// Backed by localStorage so it works without any extra database table. The
// async, { data, error } return shape matches how the components consume it
// (and mirrors the Supabase client style) so it can be swapped for a real
// table later without touching call sites.

const DATASET_TYPE_PREFERENCE_KEY = 'datasetTypePreference'

export class DatabaseService {
  /** Return the user's last-selected dataset type (numeric/text/image/mixed). */
  static async getUserDatasetTypePreference(): Promise<{ data: string | null; error: any }> {
    try {
      const data =
        typeof localStorage !== 'undefined'
          ? localStorage.getItem(DATASET_TYPE_PREFERENCE_KEY)
          : null
      return { data, error: null }
    } catch (error) {
      return { data: null, error }
    }
  }

  /** Persist the user's selected dataset type. */
  static async saveDatasetTypePreference(datasetType: string): Promise<{ data: string | null; error: any }> {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(DATASET_TYPE_PREFERENCE_KEY, datasetType)
      }
      return { data: datasetType, error: null }
    } catch (error) {
      return { data: null, error }
    }
  }
}
