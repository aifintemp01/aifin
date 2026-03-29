const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface DocumentStatus {
  status: 'ready' | 'md_only' | 'not_found' | 'error';
  filename: string;
  node_count?: number;
  tree_path?: string;
}

export interface UploadResult {
  status: string;
  filename: string;
  stem: string;
  node_count: number;
  md_saved: string;
  tree_saved: string;
}

export interface QueryResult {
  answer: string;
  query: string;
  filename: string;
}

export interface DocumentListItem {
  filename: string;
  stem: string;
  node_count?: number;
  status: string;
}

export const PageIndexAPI = {
  async uploadPDF(file: File): Promise<UploadResult> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/pageindex/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async getStatus(filename: string): Promise<DocumentStatus> {
    const res = await fetch(`${API_BASE}/pageindex/status/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error('Status check failed');
    return res.json();
  },

  async query(filename: string, query: string): Promise<QueryResult> {
    const res = await fetch(`${API_BASE}/pageindex/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, query }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Query failed');
    }
    return res.json();
  },

  async listDocuments(): Promise<{ documents: DocumentListItem[] }> {
    const res = await fetch(`${API_BASE}/pageindex/documents`);
    if (!res.ok) throw new Error('Failed to list documents');
    return res.json();
  },
};
