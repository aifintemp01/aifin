import { useCallback, useEffect, useState } from 'react';
import { CheckCircle2, FileText, Loader2, X, XCircle } from 'lucide-react';

import { cancelPDF, getQueue, type PDFJob } from '@/services/pdf-api';

export function PDFQueueStatus() {
  const [jobs, setJobs] = useState<PDFJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const poll = useCallback(async () => {
    const all = await getQueue();
    setJobs(all);
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [poll]);

  // Only show queued/processing jobs + done/failed that haven't been dismissed
  const visible = jobs.filter(
    (j) => !dismissed.has(j.queue_id) && j.status !== 'cancelled',
  );

  if (visible.length === 0) return null;

  const handleCancel = async (queueId: string) => {
    await cancelPDF(queueId);
    await poll();
  };

  const handleDismiss = (queueId: string) => {
    setDismissed((prev) => new Set([...prev, queueId]));
  };

  return (
    <div className="fixed top-12 right-3 z-50 flex flex-col gap-1.5" style={{ maxWidth: 272 }}>
      {visible.map((job) => (
        <div
          key={job.queue_id}
          className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2 text-xs shadow-sm"
        >
          {/* Status icon */}
          {job.status === 'queued' && (
            <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          )}
          {job.status === 'processing' && (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-blue-500" />
          )}
          {job.status === 'done' && (
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-green-500" />
          )}
          {job.status === 'failed' && (
            <XCircle className="h-3.5 w-3.5 shrink-0 text-red-500" />
          )}

          {/* Label */}
          <span className="min-w-0 flex-1 truncate text-foreground">
            {job.status === 'queued' && `PDF queued (pos. ${job.position})`}
            {job.status === 'processing' && 'Generating PDF…'}
            {job.status === 'done' && `PDF sent to ${job.email}`}
            {job.status === 'failed' && (job.error ? `Failed: ${job.error.slice(0, 40)}` : 'PDF generation failed')}
          </span>

          {/* Cancel (queued only) */}
          {job.status === 'queued' && (
            <button
              onClick={() => handleCancel(job.queue_id)}
              className="shrink-0 text-muted-foreground hover:text-foreground"
              title="Cancel"
            >
              <X className="h-3 w-3" />
            </button>
          )}

          {/* Dismiss (done / failed) */}
          {(job.status === 'done' || job.status === 'failed') && (
            <button
              onClick={() => handleDismiss(job.queue_id)}
              className="shrink-0 text-muted-foreground hover:text-foreground"
              title="Dismiss"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      ))}
    </div>
  );
}