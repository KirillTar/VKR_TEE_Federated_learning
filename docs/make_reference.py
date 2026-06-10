#!/usr/bin/env python3
"""
Generate reference.docx for pandoc with GOST 7.32-2017 styles.

Styles defined:
  - Page: A4, margins 30-15-20-10 mm (left-right-top-bottom)
  - Body (Normal): Times New Roman 14pt, 1.5 line spacing, first-line indent 1.25cm
  - Headings 1-3: TNR bold, appropriate sizes, spacing before/after
  - Code (Source Code): Courier New 10pt, single spacing
  - Table text: TNR 12pt
  - Figure caption / Table caption: TNR 12pt, centered
  - TOC styles
"""

from docx import Document
from docx.shared import Pt, Cm, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn


def set_style_font(style, name="Times New Roman", size=14, bold=False, italic=False, color=None):
    font = style.font
    font.name = name
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    if color:
        font.color.rgb = color
    # For Cyrillic: set East Asian and Complex Script fonts too
    rpr = style.element.get_or_add_rPr()
    for tag in [qn("w:rFonts")]:
        elem = rpr.find(tag)
        if elem is None:
            elem = rpr.makeelement(tag, {})
            rpr.insert(0, elem)
        elem.set(qn("w:ascii"), name)
        elem.set(qn("w:hAnsi"), name)
        elem.set(qn("w:cs"), name)
        elem.set(qn("w:eastAsia"), name)


def set_paragraph_format(style, alignment=None, space_before=0, space_after=0,
                         line_spacing=1.5, first_line_indent=None,
                         keep_with_next=False, page_break_before=False):
    pf = style.paragraph_format
    if alignment is not None:
        pf.alignment = alignment
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line_spacing
    if first_line_indent is not None:
        pf.first_line_indent = Cm(first_line_indent)
    pf.keep_with_next = keep_with_next
    pf.page_break_before = page_break_before


def main():
    doc = Document()

    # --- Page setup (A4, GOST margins) ---
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.orientation = WD_ORIENT.PORTRAIT
        section.left_margin = Mm(30)
        section.right_margin = Mm(15)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)
        section.header_distance = Mm(10)
        section.footer_distance = Mm(10)

    # --- Normal (body text) ---
    style = doc.styles["Normal"]
    set_style_font(style, size=14)
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                         first_line_indent=1.25, line_spacing=1.5)

    # --- Heading 1: chapter title ---
    style = doc.styles["Heading 1"]
    set_style_font(style, size=16, bold=True, color=RGBColor(0, 0, 0))
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                         space_before=24, space_after=12,
                         line_spacing=1.5, page_break_before=False,
                         keep_with_next=True)

    # --- Heading 2: section title ---
    style = doc.styles["Heading 2"]
    set_style_font(style, size=14, bold=True, color=RGBColor(0, 0, 0))
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                         space_before=18, space_after=6,
                         line_spacing=1.5, first_line_indent=1.25,
                         keep_with_next=True)

    # --- Heading 3: subsection title ---
    style = doc.styles["Heading 3"]
    set_style_font(style, size=14, bold=True, color=RGBColor(0, 0, 0))
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                         space_before=12, space_after=6,
                         line_spacing=1.5, first_line_indent=1.25,
                         keep_with_next=True)

    # --- Source Code (for code blocks) ---
    if "Source Code" not in [s.name for s in doc.styles]:
        style = doc.styles.add_style("Source Code", 1)  # paragraph style
    else:
        style = doc.styles["Source Code"]
    set_style_font(style, name="Courier New", size=10)
    set_paragraph_format(style, line_spacing=1.0, space_before=3, space_after=3)

    # --- Figure caption ---
    style = doc.styles["Caption"]
    set_style_font(style, size=12, italic=False, color=RGBColor(0, 0, 0))
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                         space_before=6, space_after=12, line_spacing=1.5)

    # --- Table Grid (table style) ---
    # Just ensure default table text is 12pt TNR
    # pandoc uses "Table" paragraph style inside tables
    if "Table" not in [s.name for s in doc.styles]:
        pass  # pandoc will create as needed

    # --- Block Quote (for notes, remarks) ---
    if "Block Text" in [s.name for s in doc.styles]:
        style = doc.styles["Block Text"]
    else:
        style = doc.styles.add_style("Block Text", 1)
    set_style_font(style, size=12, italic=True)
    set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                         first_line_indent=1.25, line_spacing=1.5)

    # --- TOC Heading ---
    if "TOC Heading" in [s.name for s in doc.styles]:
        style = doc.styles["TOC Heading"]
        set_style_font(style, size=16, bold=True, color=RGBColor(0, 0, 0))
        set_paragraph_format(style, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                             space_before=24, space_after=12, line_spacing=1.5)

    # Add a dummy paragraph so the doc is valid
    p = doc.add_paragraph("")
    p.style = doc.styles["Normal"]

    doc.save("docs/reference.docx")
    print("Created docs/reference.docx")


if __name__ == "__main__":
    main()
