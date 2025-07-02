import base64
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from typing import Dict, List, Tuple

from models import StockDigestOutput, TargetedResearch


def create_pdf_styles() -> Dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()
    
    styles = {
        'title': ParagraphStyle(
            'CustomTitle',
            parent=base_styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        ),
        'subtitle': ParagraphStyle(
            'CustomSubtitle',
            parent=base_styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkblue
        ),
        'normal': ParagraphStyle(
            'CustomNormal',
            parent=base_styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ),
        'url': ParagraphStyle(
            'URLStyle',
            parent=base_styles['Normal'],
            fontSize=8,
            spaceAfter=6,
            textColor=colors.blue,
            leftIndent=20
        )
    }
    
    return styles


def format_date(generated_at: str) -> str:
    date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
    return date.strftime("%B %d, %Y")


def add_section(story: List, styles: Dict, title: str, content: str, add_page_break: bool = False):
    if title:
        story.append(Paragraph(title, styles['subtitle']))
    story.append(Paragraph(content, styles['normal']))
    story.append(Spacer(1, 12))
    if add_page_break:
        story.append(PageBreak())


def build_financial_table(finance_data) -> Table:
    table_data = [
        ['Metric', 'Value'],
        ['Current Price', f"${finance_data.current_price:.2f}"],
    ]
    
    # Add optional fields if they exist
    optional_fields = {
        'Market Cap': (finance_data.market_cap, lambda x: f"${x:,.0f}"),
    }
    
    for label, (value, formatter) in optional_fields.items():
        if value:
            table_data.append([label, formatter(value)])
    
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


def add_insights_section(story: List, styles: Dict, insights):
    story.append(Paragraph("Key Insights", styles['subtitle']))
    
    if isinstance(insights, list):
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles['normal']))
    else:
        story.append(Paragraph(insights, styles['normal']))
    
    story.append(Spacer(1, 12))


def add_targeted_research(story: List, styles: Dict, ticker: str, targeted_research: Dict):
    if ticker not in targeted_research:
        return
    
    story.append(Paragraph("Targeted Research Summary", styles['subtitle']))
    research = targeted_research[ticker]
    
    for category, stories in research.model_dump().items():
        if category != 'ticker' and stories:
            category_title = category.replace('_', ' ').title()
            story.append(Paragraph(f"{category_title}:", styles['normal']))
            
            for story_item in stories[:2]:  # Limit to first 2 items
                title = story_item.get('title', 'No title')
                story.append(Paragraph(f"• {title}", styles['normal']))
            
            story.append(Spacer(1, 6))


def build_stock_report(story: List, styles: Dict, ticker: str, report, targeted_research: Dict):
    story.append(Paragraph(f"{ticker} - {report.company_name}", styles['title']))
    story.append(Spacer(1, 12))
    
    # Main sections
    sections = [
        ("Summary", report.summary),
        ("Current Performance", report.current_performance),
        ("Risk Assessment", report.risk_assessment),
        ("Price Outlook", report.price_outlook),
        ("Recommendation", report.recommendation)
    ]
    
    for title, content in sections:
        if title == "Summary":  # Add financial data after summary
            add_section(story, styles, title, content)
            
            if report.finance_data:
                story.append(Paragraph("Financial Data", styles['subtitle']))
                story.append(build_financial_table(report.finance_data))
                story.append(Spacer(1, 12))
            
            add_insights_section(story, styles, report.key_insights)
        else:
            add_section(story, styles, title, content)
    
    add_targeted_research(story, styles, ticker, targeted_research)
    
    story.append(PageBreak())


def build_sources_section(story: List, styles: Dict, structured_reports: StockDigestOutput):
    story.append(Paragraph("Research Sources", styles['title']))
    story.append(Spacer(1, 12))
    
    # Collect and sort all sources
    all_sources = []
    for ticker, report in structured_reports.reports.items():
        if report.sources:
            for source in report.sources:
                source_with_ticker = {**source, 'ticker': ticker}
                all_sources.append(source_with_ticker)
    
    all_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    if all_sources:
        story.append(Paragraph("The following sources were used for this analysis:", styles['normal']))
        story.append(Spacer(1, 8))
        
        for source in all_sources[:20]:  # Limit to top 20
            title = source.get('title', 'No title')
            url = source.get('url', '')
            
            story.append(Paragraph(f"• {title}", styles['normal']))
            if url:
                story.append(Paragraph(url, styles['url']))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No sources available for this analysis.", styles['normal']))


def build_recommendations_section(story: List, styles: Dict, ticker_suggestions: Dict[str, str]):
    if not ticker_suggestions:
        return
        
    story.append(Paragraph("Stock Recommendations", styles['title']))
    story.append(Paragraph("Current analyst picks and trending stocks:", styles['normal']))
    story.append(Spacer(1, 12))
    
    for ticker, reason in ticker_suggestions.items():
        story.append(Paragraph(f"<b>{ticker}</b>", styles['normal']))
        story.append(Paragraph(reason, styles['normal']))
        story.append(Spacer(1, 8))
    
    story.append(PageBreak())


def generate_pdf(structured_reports: StockDigestOutput, targeted_research: Dict) -> Tuple[str, str]:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = create_pdf_styles()
    formatted_date = format_date(structured_reports.generated_at)
    
    story = []
    
    story.append(Paragraph("Daily Stock Digest Report", styles['title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {formatted_date}", styles['normal']))
    story.append(Spacer(1, 20))
    add_section(story, styles, "Market Overview", structured_reports.market_overview, add_page_break=True)
    
    for ticker, report in structured_reports.reports.items():
        build_stock_report(story, styles, ticker, report, targeted_research)
    
    build_recommendations_section(story, styles, structured_reports.ticker_suggestions)
    build_sources_section(story, styles, structured_reports)
    
    doc.build(story)
    
    pdf_content = buffer.getvalue()
    buffer.close()
    
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    filename = f"stock_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    return pdf_base64, filename