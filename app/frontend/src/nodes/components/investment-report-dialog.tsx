import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { extractBaseAgentKey } from '@/data/node-mappings';
import { requestPDF } from '@/services/pdf-api';
import { createAgentDisplayNames } from '@/utils/text-utils';
import { ArrowDown, ArrowUp, CheckCircle2, FileDown, Mail, Minus, X } from 'lucide-react';
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface InvestmentReportDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  outputNodeData: any;
  connectedAgentIds: Set<string>;
}

type ActionType = 'long' | 'short' | 'hold';

export function InvestmentReportDialog({
  isOpen,
  onOpenChange,
  outputNodeData,
  connectedAgentIds,
}: InvestmentReportDialogProps) {
  // ── PDF state ───────────────────────────────────────────────────────────
  const [showPDFForm, setShowPDFForm] = useState(false);
  const [pdfEmail, setPdfEmail] = useState('');
  const [pdfSubmitting, setPdfSubmitting] = useState(false);
  const [pdfQueued, setPdfQueued] = useState(false);

  const handleGeneratePDF = async () => {
    if (!pdfEmail || !outputNodeData) return;
    setPdfSubmitting(true);
    try {
      await requestPDF(pdfEmail, {
        decisions: outputNodeData.decisions,
        analyst_signals: outputNodeData.analyst_signals,
        current_prices: outputNodeData.current_prices,
        tickers: Object.keys(outputNodeData.decisions || {}),
      });
      setPdfQueued(true);
      setShowPDFForm(false);
      setPdfEmail('');
    } catch (e) {
      console.error('[PDF] request failed:', e);
    } finally {
      setPdfSubmitting(false);
    }
  };

  const resetPDFState = () => {
    setShowPDFForm(false);
    setPdfEmail('');
    setPdfQueued(false);
    setPdfSubmitting(false);
  };

  // ── early returns ───────────────────────────────────────────────────────
  if (outputNodeData?.decisions?.backtest?.type === 'backtest_complete') {
    return null;
  }
  if (!outputNodeData || !outputNodeData.decisions) {
    return null;
  }

  // ── helpers ─────────────────────────────────────────────────────────────
  const getActionIcon = (action: ActionType) => {
    switch (action) {
      case 'long':  return <ArrowUp className="h-4 w-4 text-green-500" />;
      case 'short': return <ArrowDown className="h-4 w-4 text-red-500" />;
      case 'hold':  return <Minus className="h-4 w-4 text-yellow-500" />;
      default:      return null;
    }
  };

  const getSignalBadge = (signal: string) => {
    const variant =
      signal === 'bullish' ? 'success' :
      signal === 'bearish' ? 'destructive' : 'outline';
    return <Badge variant={variant as any}>{signal}</Badge>;
  };

  const getConfidenceBadge = (confidence: number) => {
    const variant = confidence >= 50 ? 'success' : confidence >= 0 ? 'warning' : 'outline';
    return <Badge variant={variant as any}>{Number(confidence.toFixed(1))}%</Badge>;
  };

  const tickers = Object.keys(outputNodeData.decisions || {});
  const connectedUniqueAgentIds = Array.from(connectedAgentIds);
  const agents = Object.keys(outputNodeData.analyst_signals || {}).filter(
    (agent) =>
      extractBaseAgentKey(agent) !== 'risk_management_agent' &&
      connectedUniqueAgentIds.includes(agent),
  );
  const agentDisplayNames = createAgentDisplayNames(agents);

  // ── render ──────────────────────────────────────────────────────────────
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) resetPDFState();
        onOpenChange(open);
      }}
    >
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
        <DialogDescription className="sr-only">
          Investment analysis report showing trading signals and analyst recommendations
        </DialogDescription>
          <DialogTitle className="text-xl font-bold">Investment Report</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 my-4">
          {/* ── Summary ──────────────────────────────────────────────── */}
          <section>
            <h2 className="text-lg font-semibold mb-4">Summary</h2>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>
                  Recommended trading actions based on analyst signals
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Price</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>Quantity</TableHead>
                      <TableHead>Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {tickers.map((ticker) => {
                      const decision = outputNodeData.decisions[ticker];
                      const currentPrice = outputNodeData.current_prices?.[ticker] || 'N/A';
                      return (
                        <TableRow key={ticker}>
                          <TableCell className="font-medium">{ticker}</TableCell>
                          <TableCell>
                            ${typeof currentPrice === 'number' ? currentPrice.toFixed(2) : currentPrice}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              {getActionIcon(decision.action as ActionType)}
                              <span className="capitalize">{decision.action}</span>
                            </div>
                          </TableCell>
                          <TableCell>{decision.quantity}</TableCell>
                          <TableCell>{getConfidenceBadge(decision.confidence)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </section>

          {/* ── PDF Report strip ─────────────────────────────────────── */}
          <div className="flex items-center gap-3 rounded-lg border border-dashed border-border px-4 py-2.5">
            {/* Idle — show button */}
            {!showPDFForm && !pdfQueued && (
              <button
                onClick={() => setShowPDFForm(true)}
                className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                <FileDown className="h-4 w-4" />
                Generate PDF Report
              </button>
            )}

            {/* Email input form */}
            {showPDFForm && !pdfQueued && (
              <>
                <Mail className="h-4 w-4 shrink-0 text-muted-foreground" />
                <input
                  type="email"
                  placeholder="your@email.com"
                  value={pdfEmail}
                  onChange={(e) => setPdfEmail(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleGeneratePDF()}
                  className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                  autoFocus
                />
                <button
                  onClick={handleGeneratePDF}
                  disabled={pdfSubmitting || !pdfEmail.includes('@')}
                  className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-40"
                >
                  {pdfSubmitting ? 'Queuing…' : 'Send PDF'}
                </button>
                <button
                  onClick={() => { setShowPDFForm(false); setPdfEmail(''); }}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              </>
            )}

            {/* Queued confirmation */}
            {pdfQueued && (
              <>
                <CheckCircle2 className="h-4 w-4 shrink-0 text-green-500" />
                <span className="flex-1 text-sm text-muted-foreground">
                  PDF queued — check your email shortly
                </span>
                <button
                  onClick={resetPDFState}
                  className="text-sm text-muted-foreground hover:text-foreground"
                >
                  Generate another
                </button>
              </>
            )}
          </div>

          {/* ── Analyst Signals ──────────────────────────────────────── */}
          <section>
            <h2 className="text-lg font-semibold mb-4">Analyst Signals</h2>
            <Accordion type="multiple" className="w-full">
              {tickers.map((ticker) => (
                <AccordionItem key={ticker} value={ticker}>
                  <AccordionTrigger className="text-base font-medium px-4 py-3 bg-muted/30 rounded-md hover:bg-muted/50">
                    <div className="flex items-center gap-2">
                      {ticker}
                      <div className="flex items-center gap-1">
                        {getActionIcon(outputNodeData.decisions[ticker].action as ActionType)}
                        <span className="text-sm font-normal text-muted-foreground">
                          {outputNodeData.decisions[ticker].action}{' '}
                          {outputNodeData.decisions[ticker].quantity} shares
                        </span>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pt-4 px-1">
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 gap-4">
                        {agents.map((agent) => {
                          const signal = outputNodeData.analyst_signals[agent]?.[ticker];
                          if (!signal) return null;
                          return (
                            <Card key={agent} className="overflow-hidden">
                              <CardHeader className="bg-muted/50 pb-3">
                                <div className="flex items-center justify-between">
                                  <CardTitle className="text-base">
                                    {agentDisplayNames.get(agent) || agent}
                                  </CardTitle>
                                  <div className="flex items-center gap-2">
                                    {getSignalBadge(signal.signal)}
                                    {getConfidenceBadge(signal.confidence)}
                                  </div>
                                </div>
                              </CardHeader>
                              <CardContent className="pt-3">
                                {typeof signal.reasoning === 'string' ? (
                                  <p className="text-sm whitespace-pre-line">{signal.reasoning}</p>
                                ) : (
                                  <div className="max-h-48 overflow-y-auto bg-muted/30">
                                    <SyntaxHighlighter
                                      language="json"
                                      style={vscDarkPlus}
                                      className="text-sm rounded-md"
                                      customStyle={{
                                        fontSize: '0.875rem',
                                        margin: 0,
                                        padding: '12px',
                                        backgroundColor: 'hsl(var(--muted))',
                                      }}
                                    >
                                      {JSON.stringify(signal.reasoning, null, 2)}
                                    </SyntaxHighlighter>
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          );
                        })}
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}