const API_BASE = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';

export interface PDFJob {
  queue_id: string;
  status: 'queued' | 'processing' | 'done' | 'failed' | 'cancelled';
  email: string;
  position: number;
  created_at: string;
  error?: string;
}

export interface PDFRunData {
  decisions: Record<string, { action: string; quantity: number; confidence: number; reasoning: any }>;
  analyst_signals: Record<string, Record<string, { signal: string; confidence: number; reasoning: any }>>;
  current_prices: Record<string, number>;
  tickers?: string[];
  flow_name?: string;
  start_date?: string;
  end_date?: string;
}

export async function requestPDF(
  email: string,
  run_data: PDFRunData,
): Promise<{ queue_id: string; position: number; message: string }> {
  const res = await fetch(`${API_BASE}/pdf/request`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, run_data }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `PDF request failed (${res.status})`);
  }
  return res.json();
}

export async function getQueue(): Promise<PDFJob[]> {
  try {
    const res = await fetch(`${API_BASE}/pdf/queue`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function cancelPDF(queue_id: string): Promise<void> {
  await fetch(`${API_BASE}/pdf/queue/${queue_id}`, { method: 'DELETE' });
}