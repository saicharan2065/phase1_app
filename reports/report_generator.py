import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class ReportGenerator:
    def __init__(self, storage_dir="storage/reports"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def generate_pdf_report(self, report_name, title, content_lines):
        filepath = os.path.join(self.storage_dir, f"{report_name}.pdf")
        c = canvas.Canvas(filepath, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, title)
        
        c.setFont("Helvetica", 12)
        y = 710
        for line in content_lines:
            # Simple wrapping for long lines
            c.drawString(50, y, str(line)[:90]) 
            y -= 20
            if y < 50:
                c.showPage()
                y = 750
        c.save()
        return filepath

    def generate_excel_report(self, report_name, dataframes_dict):
        filepath = os.path.join(self.storage_dir, f"{report_name}.xlsx")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in dataframes_dict.items():
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return filepath
