import { useState, useRef, useEffect } from 'react';
import {
  Upload, FileText, Send, Loader2,
  ChevronRight, AlertCircle, CheckCircle2, X
} from 'lucide-react';
import { PageIndexAPI, DocumentListItem, QueryResult } from '@/services/pageindex-api';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function TreeQueryPage() {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { loadDocuments(); }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function loadDocuments() {
    try {
      const res = await PageIndexAPI.listDocuments();
      setDocuments(res.documents);
    } catch {
      // silent
    }
  }

  async function handleFileUpload(file: File) {
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files are supported.');
      return;
    }
    setIsUploading(true);
    setError(null);
    setUploadProgress('Uploading PDF...');
    try {
      setUploadProgress('Running OCR — this may take a few minutes...');
      const result = await PageIndexAPI.uploadPDF(file);
      setUploadProgress(`Done — ${result.node_count} nodes indexed`);
      await loadDocuments();
      setSelectedDoc(result.filename);
      setMessages([]);
      setTimeout(() => setUploadProgress(null), 3000);
    } catch (e: any) {
      setError(e.message || 'Upload failed');
      setUploadProgress(null);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleQuery() {
    if (!input.trim() || !selectedDoc || isQuerying) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsQuerying(true);
    setError(null);

    try {
      const result: QueryResult = await PageIndexAPI.query(selectedDoc, userMsg.content);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.answer,
        timestamp: new Date(),
      }]);
    } catch (e: any) {
      setError(e.message || 'Query failed');
    } finally {
      setIsQuerying(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }

  const selectedDocInfo = documents.find(d => d.filename === selectedDoc);

  return (
    <div className="flex h-full w-full overflow-hidden bg-background text-foreground">

      {/* Left panel — document list */}
      <div className="w-60 flex-shrink-0 border-r border-border flex flex-col">
        <div className="px-4 py-3 border-b border-border">
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            Documents
          </p>
        </div>

        {/* Upload zone */}
        <div
          className={cn(
            "mx-3 mt-3 rounded-lg border-2 border-dashed p-4 text-center cursor-pointer transition-all duration-150",
            isDragging ? "border-primary bg-primary/10" : "border-border hover:border-primary/40 hover:bg-primary/5",
            isUploading && "opacity-50 pointer-events-none"
          )}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
          />
          {isUploading
            ? <Loader2 size={16} className="mx-auto mb-1.5 animate-spin text-primary" />
            : <Upload size={16} className="mx-auto mb-1.5 text-muted-foreground" />
          }
          <p className="text-xs text-muted-foreground leading-snug">
            {isUploading ? uploadProgress : 'Drop PDF or click to upload'}
          </p>
        </div>

        {uploadProgress && !isUploading && (
          <div className="mx-3 mt-2 flex items-center gap-1.5 text-xs text-green-500">
            <CheckCircle2 size={11} />
            {uploadProgress}
          </div>
        )}

        {/* List */}
        <div className="flex-1 overflow-y-auto mt-2 px-2 pb-2 space-y-0.5">
          {documents.length === 0 && !isUploading && (
            <p className="text-xs text-muted-foreground text-center mt-8 px-3 leading-relaxed">
              No documents yet.
              <br />Upload a PDF to start querying.
            </p>
          )}
          {documents.map(doc => (
            <button
              key={doc.filename}
              onClick={() => { setSelectedDoc(doc.filename); setMessages([]); setError(null); }}
              className={cn(
                "w-full text-left px-3 py-2 rounded-md flex items-start gap-2 transition-colors",
                selectedDoc === doc.filename
                  ? "bg-primary/15 text-foreground"
                  : "hover:bg-ramp-grey-700/40 text-muted-foreground hover:text-foreground"
              )}
            >
              <FileText size={12} className="mt-0.5 flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium truncate">{doc.stem}</p>
                {doc.node_count && (
                  <p className="text-[10px] text-muted-foreground">{doc.node_count} nodes</p>
                )}
              </div>
              {selectedDoc === doc.filename && (
                <ChevronRight size={11} className="flex-shrink-0 mt-0.5" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Right panel — chat */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Header */}
        <div className="px-5 py-2.5 border-b border-border flex items-center gap-2.5 h-10">
          {selectedDocInfo ? (
            <>
              <FileText size={13} className="text-primary flex-shrink-0" />
              <p className="text-sm font-medium truncate flex-1">{selectedDocInfo.stem}</p>
              {selectedDocInfo.node_count && (
                <p className="text-xs text-muted-foreground flex-shrink-0">
                  {selectedDocInfo.node_count} nodes
                </p>
              )}
              <button
                onClick={() => { setSelectedDoc(null); setMessages([]); }}
                className="text-muted-foreground hover:text-foreground transition-colors ml-1"
              >
                <X size={13} />
              </button>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Select a document to start querying</p>
          )}
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">

          {/* Empty state — no doc selected */}
          {!selectedDoc && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                <FileText size={20} className="text-primary" />
              </div>
              <p className="text-sm text-muted-foreground max-w-xs">
                Upload a PDF on the left, then select it to ask questions about its contents.
              </p>
            </div>
          )}

          {/* Empty state — doc selected but no messages */}
          {selectedDoc && messages.length === 0 && !isQuerying && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <p className="text-sm text-muted-foreground">
                Ask anything about{' '}
                <span className="text-foreground font-medium">{selectedDocInfo?.stem}</span>
              </p>
              <div className="flex flex-wrap gap-2 justify-center mt-1">
                {[
                  'What are the main business segments?',
                  'Summarise the key financial highlights',
                  'What are the major risk factors?',
                ].map(s => (
                  <button
                    key={s}
                    onClick={() => { setInput(s); textareaRef.current?.focus(); }}
                    className="text-xs px-3 py-1.5 rounded-full border border-border hover:border-primary/40 hover:bg-primary/5 text-muted-foreground hover:text-foreground transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map(msg => (
            <div
              key={msg.id}
              className={cn("flex gap-2.5", msg.role === 'user' ? "justify-end" : "justify-start")}
            >
              {msg.role === 'assistant' && (
                <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <FileText size={11} className="text-primary" />
                </div>
              )}
              <div className={cn(
                "max-w-[75%] rounded-xl px-4 py-2.5 text-sm leading-relaxed",
                msg.role === 'user'
                  ? "bg-primary/20 text-foreground rounded-br-sm"
                  : "bg-panel border border-border text-foreground rounded-bl-sm"
              )}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
                <p className="text-[10px] text-muted-foreground mt-1">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}

          {/* Thinking indicator */}
          {isQuerying && (
            <div className="flex gap-2.5 justify-start">
              <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Loader2 size={11} className="text-primary animate-spin" />
              </div>
              <div className="bg-panel border border-border rounded-xl rounded-bl-sm px-4 py-3">
                <div className="flex gap-1 items-center">
                  <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2.5">
              <AlertCircle size={13} className="flex-shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input bar */}
        <div className="px-4 py-3 border-t border-border">
          <div className={cn(
            "flex items-end gap-2 rounded-xl border px-3 py-2 transition-colors",
            selectedDoc
              ? "border-border bg-panel/50 focus-within:border-primary/40"
              : "border-border/30 bg-panel/20 opacity-50"
          )}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!selectedDoc || isQuerying}
              placeholder={
                selectedDoc
                  ? 'Ask a question about the document… (Enter to send)'
                  : 'Select a document first'
              }
              rows={1}
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground resize-none outline-none min-h-[24px] max-h-[120px] leading-6"
            />
            <button
              onClick={handleQuery}
              disabled={!input.trim() || !selectedDoc || isQuerying}
              className={cn(
                "flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center transition-all",
                input.trim() && selectedDoc && !isQuerying
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              {isQuerying
                ? <Loader2 size={13} className="animate-spin" />
                : <Send size={13} />
              }
            </button>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1.5 px-1">
            Shift+Enter for new line · Enter to send
          </p>
        </div>
      </div>
    </div>
  );
}
