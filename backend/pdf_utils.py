"""
PDF generation utilities for stock digest reports
"""

import base64
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from typing import Dict, List, Tuple

from models import StockDigestOutput, TargetedResearch


def create_pdf_styles():
    """Create and return custom PDF styles"""
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    url_style = ParagraphStyle(
        'URLStyle',
        parent=normal_style,
        fontSize=8,
        textColor=colors.blue,
        leftIndent=20
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'normal': normal_style,
        'url': url_style
    }


def format_date(generated_at: str) -> str:
    """Format the generated date nicely"""
    generated_date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
    return generated_date.strftime("%B %d, %Y")


def build_title_page(story: List, styles: Dict, formatted_date: str):
    """Build the title page content"""
    story.append(Paragraph("Daily Stock Digest Report", styles['title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {formatted_date}", styles['normal']))
    story.append(Spacer(1, 20))


def build_market_overview(story: List, styles: Dict, market_overview: str):
    """Build the market overview section"""
    story.append(Paragraph("Market Overview", styles['subtitle']))
    story.append(Paragraph(market_overview, styles['normal']))
    story.append(PageBreak())


def build_financial_table(finance_data) -> Table:
    """Build the financial data table"""
    table_data = [
        ['Metric', 'Value'],
        ['Current Price', f"${finance_data.current_price:.2f}"],
        ['Previous Close', f"${finance_data.previous_close:.2f}"],
        ['Change %', f"{finance_data.change_percent:.2f}%"],
        ['Volume', f"{finance_data.volume:,}"],
    ]
    
    if finance_data.market_cap:
        table_data.append(['Market Cap', f"${finance_data.market_cap:,.0f}"])
    if finance_data.pe_ratio:
        table_data.append(['P/E Ratio', f"{finance_data.pe_ratio:.2f}"])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    return table


def build_stock_report(story: List, styles: Dict, ticker: str, report, targeted_research: Dict):
    """Build individual stock report content"""
    story.append(Paragraph(f"{ticker} - {report.company_name}", styles['title']))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph("Research Summary", styles['subtitle']))
    story.append(Paragraph(report.summary, styles['normal']))
    story.append(Spacer(1, 12))
    

    
    # Financial Data
    if report.finance_data:
        story.append(Paragraph("Financial Data", styles['subtitle']))
        finance_table = build_financial_table(report.finance_data)
        story.append(finance_table)
        story.append(Spacer(1, 12))
    
    # Key Insights
    story.append(Paragraph("Key Insights", styles['subtitle']))
    if isinstance(report.key_insights, list):
        for insight in report.key_insights:
            story.append(Paragraph(f"• {insight}", styles['normal']))
    else:
        story.append(Paragraph(report.key_insights, styles['normal']))
    story.append(Spacer(1, 12))
    
    # Current Performance
    story.append(Paragraph("Current Performance", styles['subtitle']))
    story.append(Paragraph(report.current_performance, styles['normal']))
    story.append(Spacer(1, 12))
    
    # Risk Assessment
    story.append(Paragraph("Risk Assessment", styles['subtitle']))
    story.append(Paragraph(report.risk_assessment, styles['normal']))
    story.append(Spacer(1, 12))
    
    # Price Outlook
    story.append(Paragraph("Price Outlook", styles['subtitle']))
    story.append(Paragraph(report.price_outlook, styles['normal']))
    story.append(Spacer(1, 12))
    
    # Recommendation
    story.append(Paragraph("Recommendation", styles['subtitle']))
    story.append(Paragraph(report.recommendation, styles['normal']))
    story.append(Spacer(1, 12))
    
    # Targeted Research Summary
    if ticker in targeted_research:
        story.append(Paragraph("Targeted Research Summary", styles['subtitle']))
        research = targeted_research[ticker]
        
        for category, stories in research.model_dump().items():
            if category != 'ticker' and stories:
                story.append(Paragraph(f"{category.replace('_', ' ').title()}:", styles['normal']))
                for story_item in stories[:2]:
                    story.append(Paragraph(f"• {story_item.get('title', 'No title')}", styles['normal']))
                story.append(Spacer(1, 6))
    
    story.append(PageBreak())


def build_sources_section(story: List, styles: Dict, structured_reports: StockDigestOutput):
    """Build the sources section"""
    story.append(Paragraph("Research Sources", styles['title']))
    story.append(Spacer(1, 12))
    
    # Collect all sources from all reports
    all_sources = []
    for ticker, report in structured_reports.reports.items():
        if report.sources:
            for source in report.sources:
                source_with_ticker = source.copy()
                source_with_ticker['ticker'] = ticker
                all_sources.append(source_with_ticker)
    
    # Sort sources by relevance score
    all_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    if all_sources:
        story.append(Paragraph("The following sources were used for this analysis:", styles['normal']))
        story.append(Spacer(1, 8))
        
        for source in all_sources[:20]:
            title = source.get('title', 'No title')
            url = source.get('url', '')
            
            # Add title
            story.append(Paragraph(f"• {title}", styles['normal']))
            
            # Add URL if available
            if url:
                story.append(Paragraph(url, styles['url']))
            
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No sources available for this analysis.", styles['normal']))


def generate_pdf(structured_reports: StockDigestOutput, targeted_research: Dict) -> Tuple[str, str]:
    """
    Generate PDF report from structured data
    
    Returns:
        Tuple of (pdf_base64_string, filename)
    """
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Get styles
    styles = create_pdf_styles()
    
    # Format date
    formatted_date = format_date(structured_reports.generated_at)
    
    # Build PDF content
    story = []
    
    # Title page
    build_title_page(story, styles, formatted_date)
    
    # Market Overview
    build_market_overview(story, styles, structured_reports.market_overview)
    
    # Individual Stock Reports
    for ticker, report in structured_reports.reports.items():
        build_stock_report(story, styles, ticker, report, targeted_research)
    
    # Sources Section
    build_sources_section(story, styles, structured_reports)
    
    # Build PDF
    doc.build(story)
    
    # Get PDF content and encode to base64
    pdf_content = buffer.getvalue()
    buffer.close()
    
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    filename = f"stock_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    return pdf_base64, filename 