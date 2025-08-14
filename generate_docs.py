import os
from fpdf import FPDF
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, PythonLexer, YamlLexer, JsonLexer, TextLexer
from pygments.formatters import ImageFormatter
import base64
from io import BytesIO
from PIL import Image
import re

class PDFGenerator:
    def __init__(self, output_path="code_documentation.pdf"):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.output_path = output_path
        self.setup_fonts()
        self.current_y = 10
        self.page_width = self.pdf.w - 20  # 10mm margins on each side
        
    def setup_fonts(self):
        # Using standard PDF fonts that don't require external files
        self.pdf.set_font('Courier', '', 8)
        
    def add_header(self, text, level=1):
        self.pdf.ln(10)
        if level == 1:
            self.pdf.set_font('Courier', 'B', 16)
            self.pdf.cell(0, 10, text, ln=True, border=0)
            self.pdf.line(10, self.pdf.get_y(), self.page_width + 10, self.pdf.get_y())
            self.pdf.ln(5)
        else:
            self.pdf.set_font('Courier', 'B', 12)
            self.pdf.cell(0, 8, text, ln=True, border=0)
        self.pdf.set_font('Courier', '', 8)
        
    def add_code(self, code, lexer):
        try:
            # Format code with line numbers
            lines = code.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines, 1):
                formatted_lines.append(f"{i:4d} | {line}")
            code_with_numbers = '\n'.join(formatted_lines)
            
            # Add to PDF
            self.pdf.multi_cell(0, 4, code_with_numbers)
            self.pdf.ln(5)
            
        except Exception as e:
            print(f"Error processing code block: {str(e)}")
            self.pdf.multi_cell(0, 4, code)
            self.pdf.ln(5)
    
    def process_directory(self, base_path, exclude_dirs=None):
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', '.ipynb_checkpoints', '.idea', 'venv', 'env']
            
        for root, dirs, files in os.walk(base_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.startswith('.') or file.endswith(('.pyc', '.pyo', '.pyd', '.so', '.dll')):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(base_path))
                
                try:
                    # Skip large files
                    if os.path.getsize(file_path) > 5 * 1024 * 1024:  # 5MB
                        print(f"Skipping large file: {file_path}")
                        continue
                        
                    # Determine file type and get appropriate lexer
                    if file.endswith(('.py', '.ipynb')):
                        lexer = 'python'
                    elif file.endswith(('.yaml', '.yml')):
                        lexer = 'yaml'
                    elif file.endswith('.json'):
                        lexer = 'json'
                    else:
                        lexer = 'text'
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Add to PDF
                    self.add_header(f"File: {rel_path}", level=2)
                    self.add_code(content, lexer)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
    
    def generate_pdf(self, directories, output_path=None):
        if output_path:
            self.output_path = output_path
            
        self.pdf.add_page()
        self.add_header("Code Documentation", level=1)
        
        for directory in directories:
            if os.path.exists(directory):
                print(f"Processing directory: {directory}")
                self.process_directory(directory)
            else:
                print(f"Directory not found: {directory}")
                
        self.pdf.output(self.output_path)
        print(f"\nPDF generated successfully at: {os.path.abspath(self.output_path)}")

# List of directories to include
directories = [
    "/Users/arkajyotisaha/Desktop/My-Thesis/code/src",
    "/Users/arkajyotisaha/Desktop/My-Thesis/code/configs",
    "/Users/arkajyotisaha/Desktop/My-Thesis/code/notebooks"
]

# Generate the PDF
print("Starting PDF generation...")
pdf_generator = PDFGenerator()
pdf_generator.generate_pdf(directories)
print("Done!")