import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    @staticmethod
    def send_report_email(
        recipients: List[str], 
        subject: str, 
        body: str, 
        pdf_content: Optional[bytes] = None
    ) -> bool:
        """
        Send an email using SMTP configuration from environment variables.
        Required ENV vars:
        - SMTP_SERVER (e.g., smtp.gmail.com)
        - SMTP_PORT (e.g., 587)
        - SMTP_USER (your email)
        - SMTP_PASSWORD (app password)
        """
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not all([smtp_server, smtp_user, smtp_password]):
            print("Email service not configured. Check environment variables.")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            if pdf_content:
                part = MIMEApplication(pdf_content, Name="report.pdf")
                part['Content-Disposition'] = 'attachment; filename="report.pdf"'
                msg.attach(part)

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
