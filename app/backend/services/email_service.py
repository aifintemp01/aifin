"""Gmail SMTP email service.

Uses GMAIL_SENDER and GMAIL_APP_PASSWORD env vars.
SMTP is synchronous — wrapped in run_in_executor so it doesn't block the event loop.
"""
import asyncio
import os
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


async def send_pdf_email(
    to_email: str,
    pdf_bytes: bytes,
    flow_name: str = "AI Hedge Fund",
) -> None:
    """Send the PDF report as an email attachment. Non-blocking."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_sync, to_email, pdf_bytes, flow_name)


def _send_sync(to_email: str, pdf_bytes: bytes, flow_name: str) -> None:
    sender = os.getenv("GMAIL_SENDER", "aifintemp01@gmail.com")
    password = os.getenv("GMAIL_APP_PASSWORD", "")

    if not password:
        raise ValueError(
            "GMAIL_APP_PASSWORD env var is not set. "
            "Generate one at myaccount.google.com → Security → App passwords."
        )

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"ai_hedge_fund_report_{date_str}.pdf"

    msg = MIMEMultipart()
    msg["From"] = f"AI Hedge Fund <{sender}>"
    msg["To"] = to_email
    msg["Subject"] = f"Investment Report — {flow_name} ({date_str})"

    body = (
        f"Hello,\n\n"
        f'Your AI Hedge Fund investment report for "{flow_name}" is attached.\n\n'
        f"The report includes:\n"
        f"  • Executive summary with trading recommendations\n"
        f"  • Portfolio allocation breakdown\n"
        f"  • Per-ticker analysis with analyst signals and reasoning\n"
        f"  • 90-day price history charts\n\n"
        f"This report is for informational purposes only and does not constitute financial advice.\n\n"
        f"— AI Hedge Fund Platform\n"
    )
    msg.attach(MIMEText(body, "plain"))

    attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    attachment.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
        print(f"[email_service] PDF sent to {to_email}")