import pandas as pd
from fpdf import FPDF

# Load your CSV (assuming it's named "participant_report.csv")
def csv_to_pdf(report):
    df = pd.read_csv(report)

# Create PDFS
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Participant Attention Report", ln=True, align='C')
    pdf.ln(10)

    # Set up column headers
    pdf.set_font("Arial", 'B', 12)
    col_widths = [40, 30, 30, 30, 50]  # Adjust column widths as needed
    columns = df.columns.tolist()

    # Add table header
    for i, col in enumerate(columns):
        pdf.cell(col_widths[i], 10, col, border=1, align='C')
    pdf.ln()

# Add table rows
    pdf.set_font("Arial", '', 11)
    for _, row in df.iterrows():
        for i, col in enumerate(columns):
            text = str(row[col])
            pdf.cell(col_widths[i], 10, text, border=1, align='C')
        pdf.ln()

# Save the PDF
    pdf.output("participant_report.pdf")
    print("PDF saved as participant_report.pdf")
