import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import unicodedata
import re

# HWPX file path constants
XML_FILENAME_HEADER = 'Contents/header.xml'
XML_FILENAME_SETTINGS = 'settings.xml'
XML_FILENAME_CONTENT_HPF = 'Contents/content.hpf'
XML_FILENAME_VERSION = 'version.xml'

# HWPX namespaces
HWPX_NAMESPACES = {
    'hh': 'http://www.hancom.co.kr/hwpml/2011/head',
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp10': 'http://www.hancom.co.kr/hwpml/2016/paragraph',
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
    'ha': 'http://www.hancom.co.kr/hwpml/2011/app',
    'hhs': 'http://www.hancom.co.kr/hwpml/2011/history',
    'hm': 'http://www.hancom.co.kr/hwpml/2011/master-page',
    'hpf': 'http://www.hancom.co.kr/schema/2011/hpf',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'opf': 'http://www.idpf.org/2007/opf/',
    'ooxmlchart': 'http://www.hancom.co.kr/hwpml/2016/ooxmlchart',
    'hwpunitchar': 'http://www.hancom.co.kr/hwpml/2016/HwpUnitChar',
    'epub': 'http://www.idpf.org/2007/ops',
    'config': 'urn:oasis:names:tc:opendocument:xmlns:config:1.0',
}


def extract_namespaces(xml_bytes: BytesIO) -> Dict[str, str]:
    """Extract namespaces from XML content"""
    namespaces = {}
    try:
        for event, elem in ET.iterparse(xml_bytes, events=['start-ns']):
            prefix, uri = elem
            if prefix:
                namespaces[prefix] = uri
    except:
        pass
    return namespaces if namespaces else HWPX_NAMESPACES


def get_text_recursive(element) -> str:
    """Recursively get all text from an element and its children"""
    text_parts = []
    if element.text:
        text_parts.append(element.text)
    for child in element:
        text_parts.append(get_text_recursive(child))
        if child.tail:
            text_parts.append(child.tail)
    return ''.join(text_parts)


@dataclass
class CaretPosition:
    list_id: int = 0
    para_id: int = 0
    char_pos: int = 0

    def __str__(self) -> str:
        return f"CaretPosition(list_id={self.list_id}, para_id={self.para_id}, char_pos={self.char_pos})"


@dataclass
class BinaryDataItem:
    index: int = 0
    item_id: str = ""
    href: str = ""
    media_type: str = ""
    is_embedded: bool = False

    def __str__(self) -> str:
        embedded = "Embedded" if self.is_embedded else "Linked"
        return f"  [{self.index}] {embedded}: ID='{self.item_id}', Path='{self.href}', Type='{self.media_type}'"


@dataclass
class FontItem:
    index: int = 0
    face: str = ""
    font_type: str = ""
    is_embedded: bool = False
    family_type: str = ""
    lang: str = ""

    def __str__(self) -> str:
        return f"  [{self.index}] '{self.face}' ({self.lang}, {self.font_type}, embedded={self.is_embedded})"


@dataclass
class Paragraph:
    index: int = 0
    text: str = ""
    style_id: str = ""
    para_id: str = ""

    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        preview = preview.replace('\n', '\\n').replace('\r', '\\r')
        return f"  [{self.index}] \"{preview}\""


@dataclass
class CharProperty:
    index: int = 0
    char_id: str = ""
    height: str = ""
    font_refs: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"  [{self.index}] ID={self.char_id}, Height={self.height}"


@dataclass
class Style:
    index: int = 0
    style_id: str = ""
    name: str = ""
    style_type: str = ""

    def __str__(self) -> str:
        return f"  [{self.index}] ID={self.style_id}, Name='{self.name}', Type={self.style_type}"


@dataclass
class Section:
    index: int = 0
    paragraphs: List[Paragraph] = field(default_factory=list)
    raw_xml: str = ""

    def get_full_text(self) -> str:
        """Get all text from this section"""
        return "\n".join(p.text for p in self.paragraphs if p.text)

    def to_dict(self) -> Dict[str, Any]:
        """Return section data as structured dictionary"""
        text = self.get_full_text()
        return {
            'section_type': 'bodytext',
            'section_title': f'Section {self.index}',
            'text': text,
            'section_order': self.index,
            'word_count': len(text.split()) if text else 0,
            'char_count': len(text) if text else 0
        }


@dataclass
class Document:
    # Document properties
    sectionCount: int = 0
    pageStartNum: int = 0
    footnoteStartNum: int = 0
    endnoteStartNum: int = 0
    pictureStartNum: int = 0
    tableStartNum: int = 0
    equationStartNum: int = 0
    caretPos: CaretPosition = field(default_factory=CaretPosition)
    binaryDataCount: int = 0
    hangulFontDataCount: int = 0
    englishFontDataCount: int = 0

    # Extracted data storage
    binary_items: List[BinaryDataItem] = field(default_factory=list)
    hangul_fonts: List[FontItem] = field(default_factory=list)
    english_fonts: List[FontItem] = field(default_factory=list)
    other_fonts: List[FontItem] = field(default_factory=list)
    char_properties: List[CharProperty] = field(default_factory=list)
    styles: List[Style] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)

    # File info
    file_list: List[str] = field(default_factory=list)
    version_info: Dict[str, str] = field(default_factory=dict)
    namespaces: Dict[str, str] = field(default_factory=dict)

    # Monitoring flag
    verbose: bool = False

    @classmethod
    def read_hwpx_document(cls, file_path: str, verbose: bool = False) -> Optional['Document']:
        """Read HWPX file (ZIP-based XML format)"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Reading HWPX file: {file_path}")
            print(f"{'='*70}")

        try:
            zipf = zipfile.ZipFile(file_path, 'r')
        except Exception as e:
            if verbose:
                print(f"[Error] Failed to open HWPX file: {e}")
            return None

        doc = cls(verbose=verbose)

        # Store file list
        doc.file_list = zipf.namelist()

        if verbose:
            print(f"\n[HWPX Archive Contents] ({len(doc.file_list)} files)")
            for f in doc.file_list:
                print(f"  - {f}")

        # Read version.xml
        doc._read_version(zipf)

        # Read header.xml (document properties, fonts, styles)
        if not doc._read_header(zipf):
            if verbose:
                print("[Error] Failed to read header.xml")
            zipf.close()
            return None

        # Read settings.xml (caret position, etc.)
        doc._read_settings(zipf)

        # Read content.hpf (binary data list)
        doc._read_content_hpf(zipf)

        # Read all section files
        doc._read_sections(zipf)

        zipf.close()
        return doc

    def _read_version(self, zipf: zipfile.ZipFile) -> bool:
        """Read version.xml"""
        if XML_FILENAME_VERSION not in zipf.namelist():
            return False

        try:
            xml_content = zipf.read(XML_FILENAME_VERSION)
            root = ET.fromstring(xml_content)

            self.version_info['version'] = root.get('version', 'N/A')

            # Get application info
            for child in root:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag_name == 'application':
                    self.version_info['app_version'] = child.get('version', '')

            if self.verbose:
                print(f"\n[Version Info]")
                for k, v in self.version_info.items():
                    print(f"  {k}: {v}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to read version.xml: {e}")
            return False

    def _read_header(self, zipf: zipfile.ZipFile) -> bool:
        """Read Contents/header.xml"""
        if XML_FILENAME_HEADER not in zipf.namelist():
            if self.verbose:
                print(f"[Error] {XML_FILENAME_HEADER} not found")
            return False

        try:
            # Extract namespaces
            xml_bytes = BytesIO(zipf.read(XML_FILENAME_HEADER))
            self.namespaces = extract_namespaces(xml_bytes)

            # Parse XML
            xml_content = zipf.read(XML_FILENAME_HEADER)
            root = ET.fromstring(xml_content)

            # Get section count from root attribute
            self.sectionCount = int(root.get('secCnt', '0'))

            if self.verbose:
                print(f"\n[Parsing header.xml]")
                print(f"  Section Count: {self.sectionCount}")

            # Parse beginNum
            self._parse_begin_num(root)

            # Parse refList (fonts, styles, char properties)
            self._parse_ref_list(root)

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Error] Failed to parse header.xml: {e}")
            return False

    def _parse_begin_num(self, root: ET.Element) -> None:
        """Parse beginNum element for starting numbers"""
        # Try with namespace
        begin_num = root.find('.//hh:beginNum', HWPX_NAMESPACES)

        # Try without namespace
        if begin_num is None:
            for elem in root.iter():
                if 'beginNum' in elem.tag:
                    begin_num = elem
                    break

        if begin_num is not None:
            self.pageStartNum = int(begin_num.get('page', '1'))
            self.footnoteStartNum = int(begin_num.get('footnote', '1'))
            self.endnoteStartNum = int(begin_num.get('endnote', '1'))
            self.pictureStartNum = int(begin_num.get('pic', '1'))
            self.tableStartNum = int(begin_num.get('tbl', '1'))
            self.equationStartNum = int(begin_num.get('equation', '1'))

            if self.verbose:
                print(f"  Page Start: {self.pageStartNum}")
                print(f"  Footnote Start: {self.footnoteStartNum}")
                print(f"  Endnote Start: {self.endnoteStartNum}")
                print(f"  Picture Start: {self.pictureStartNum}")
                print(f"  Table Start: {self.tableStartNum}")
                print(f"  Equation Start: {self.equationStartNum}")

    def _parse_ref_list(self, root: ET.Element) -> None:
        """Parse refList for fonts, char properties, and styles"""
        # Find refList
        ref_list = None
        for elem in root.iter():
            if 'refList' in elem.tag:
                ref_list = elem
                break

        if ref_list is None:
            return

        # Parse fontfaces
        for fontfaces in ref_list.iter():
            if 'fontfaces' in fontfaces.tag:
                self._parse_fontfaces(fontfaces)

        # Parse charProperties
        for char_props in ref_list.iter():
            if 'charProperties' in char_props.tag:
                self._parse_char_properties(char_props)

        # Parse styles
        for styles in ref_list.iter():
            if 'styles' in styles.tag and 'paraProperties' not in styles.tag:
                self._parse_styles(styles)

    def _parse_fontfaces(self, fontfaces: ET.Element) -> None:
        """Parse fontfaces element"""
        font_idx = 0
        for fontface in fontfaces:
            if 'fontface' in fontface.tag:
                lang = fontface.get('lang', '')
                font_count = int(fontface.get('fontCnt', '0'))

                for font in fontface:
                    if 'font' in font.tag and 'fontface' not in font.tag:
                        item = FontItem(
                            index=font_idx,
                            face=font.get('face', ''),
                            font_type=font.get('type', ''),
                            is_embedded=font.get('isEmbedded', '0') == '1',
                            lang=lang
                        )

                        # Get family type from typeInfo
                        for type_info in font:
                            if 'typeInfo' in type_info.tag:
                                item.family_type = type_info.get('familyType', '')

                        if lang == 'HANGUL':
                            self.hangul_fonts.append(item)
                        elif lang == 'LATIN':
                            self.english_fonts.append(item)
                        else:
                            self.other_fonts.append(item)

                        font_idx += 1

                if lang == 'HANGUL':
                    self.hangulFontDataCount = font_count
                elif lang == 'LATIN':
                    self.englishFontDataCount = font_count

        if self.verbose:
            print(f"  Hangul Font Count: {self.hangulFontDataCount}")
            print(f"  English Font Count: {self.englishFontDataCount}")

    def _parse_char_properties(self, char_props: ET.Element) -> None:
        """Parse character properties"""
        idx = 0
        for prop in char_props:
            if 'charPr' in prop.tag:
                item = CharProperty(
                    index=idx,
                    char_id=prop.get('id', ''),
                    height=prop.get('height', '')
                )
                self.char_properties.append(item)
                idx += 1

    def _parse_styles(self, styles_elem: ET.Element) -> None:
        """Parse styles"""
        idx = 0
        for style in styles_elem:
            if 'style' in style.tag:
                item = Style(
                    index=idx,
                    style_id=style.get('id', ''),
                    name=style.get('name', ''),
                    style_type=style.get('type', '')
                )
                self.styles.append(item)
                idx += 1

    def _read_settings(self, zipf: zipfile.ZipFile) -> bool:
        """Read settings.xml for caret position"""
        if XML_FILENAME_SETTINGS not in zipf.namelist():
            return False

        try:
            xml_content = zipf.read(XML_FILENAME_SETTINGS)
            root = ET.fromstring(xml_content)

            # Find CaretPosition
            for elem in root.iter():
                if 'CaretPosition' in elem.tag:
                    list_id = int(elem.get('listIDRef', '0'))
                    para_id = int(elem.get('paraIDRef', '0'))
                    char_pos = int(elem.get('pos', '0'))
                    self.caretPos = CaretPosition(list_id, para_id, char_pos)

                    if self.verbose:
                        print(f"\n[Settings]")
                        print(f"  Caret Position: {self.caretPos}")
                    break

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to read settings.xml: {e}")
            return False

    def _read_content_hpf(self, zipf: zipfile.ZipFile) -> bool:
        """Read Contents/content.hpf for binary data list"""
        if XML_FILENAME_CONTENT_HPF not in zipf.namelist():
            return False

        try:
            xml_content = zipf.read(XML_FILENAME_CONTENT_HPF)
            root = ET.fromstring(xml_content)

            # Find manifest
            idx = 0
            for elem in root.iter():
                if 'item' in elem.tag:
                    href = elem.get('href', '')

                    # Check if it's binary data
                    if href.startswith('BinData/'):
                        item = BinaryDataItem(
                            index=idx,
                            item_id=elem.get('id', ''),
                            href=href,
                            media_type=elem.get('media-type', ''),
                            is_embedded=elem.get('isEmbeded', '0') == '1'
                        )
                        self.binary_items.append(item)
                        idx += 1

            self.binaryDataCount = len(self.binary_items)

            if self.verbose:
                print(f"\n[Content HPF]")
                print(f"  Binary Data Count: {self.binaryDataCount}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to read content.hpf: {e}")
            return False

    def _read_sections(self, zipf: zipfile.ZipFile) -> None:
        """Read all section XML files"""
        section_idx = 0

        while True:
            section_path = f'Contents/section{section_idx}.xml'
            if section_path not in zipf.namelist():
                break

            if self.verbose:
                print(f"\n[Reading {section_path}]")

            try:
                xml_content = zipf.read(section_path)
                section = Section(index=section_idx)
                section.raw_xml = xml_content.decode('utf-8', errors='replace')

                # Parse section XML
                root = ET.fromstring(xml_content)
                self._parse_section_paragraphs(root, section)

                self.sections.append(section)

                if self.verbose:
                    print(f"  Paragraphs found: {len(section.paragraphs)}")

            except Exception as e:
                if self.verbose:
                    print(f"  [Error] Failed to parse section: {e}")

            section_idx += 1

        if self.verbose:
            print(f"\n[Total sections read: {len(self.sections)}]")

    def _parse_section_paragraphs(self, root: ET.Element, section: Section) -> None:
        """Parse paragraphs from section XML"""
        para_idx = 0

        # Find all paragraph elements (p)
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag_name == 'p':
                para = Paragraph(index=para_idx)
                para.style_id = elem.get('styleIDRef', '')
                para.para_id = elem.get('id', '')

                # Extract text from all text runs
                text_parts = []
                for child in elem.iter():
                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

                    # Handle different text elements
                    if child_tag == 't':  # Text run
                        if child.text:
                            text_parts.append(child.text)
                    elif child_tag == 'tab':  # Tab
                        text_parts.append('\t')
                    elif child_tag == 'lineBreak':  # Line break
                        text_parts.append('\n')
                    elif child_tag == 'secPr':  # Section property (ignore)
                        continue

                para.text = ''.join(text_parts)

                # Only add if there's content or it's an empty paragraph marker
                if para.text or len(list(elem)) > 0:
                    section.paragraphs.append(para)
                    para_idx += 1

    def get_full_text(self) -> str:
        """Get all text from the document"""
        texts = []
        for section in self.sections:
            texts.append(section.get_full_text())
        return "\n\n".join(texts)

    def get_sections_data(self) -> List[Dict[str, Any]]:
        """Return all sections as list of structured dictionaries"""
        return [section.to_dict() for section in self.sections]

    def extract(self) -> Dict[str, Any]:
        """Return full extraction data as dictionary"""
        full_text = self.get_full_text()

        sections = []
        for i, section in enumerate(self.sections, 1):
            text = section.get_full_text()
            sections.append({
                'section_type': 'bodytext',
                'section_title': f'Section {i}',
                'text': text,
                'section_order': i,
                'word_count': len(text.split()) if text else 0,
                'char_count': len(text) if text else 0
            })

        return {
            'text': full_text,
            'method': 'zipfile+xml',
            'sections': sections,
            'word_count': len(full_text.split()) if full_text else 0,
            'char_count': len(full_text) if full_text else 0
        }

    def clean_text(self, remove_chinese: bool = True, remove_control: bool = True) -> str:
        """Get cleaned text with optional Chinese and control character removal"""
        text = self.get_full_text()

        if remove_chinese:
            text = re.sub(r'[\u4e00-\u9fff]+', '', text)

        if remove_control:
            text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

        return text

    def print_summary(self) -> None:
        """Print a summary of all extracted data"""
        print(f"\n{'='*70}")
        print("HWPX DOCUMENT FULL EXTRACTION")
        print(f"{'='*70}")

        # Version Info
        if self.version_info:
            print(f"\n[Version Info]")
            for k, v in self.version_info.items():
                print(f"  {k}: {v}")

        # File List
        print(f"\n[Archive Contents] ({len(self.file_list)} files)")
        for f in self.file_list:
            print(f"  - {f}")

        # Document Properties
        print(f"\n[Document Properties]")
        print(f"  Section Count: {self.sectionCount}")
        print(f"  Page Start Num: {self.pageStartNum}")
        print(f"  Footnote Start Num: {self.footnoteStartNum}")
        print(f"  Endnote Start Num: {self.endnoteStartNum}")
        print(f"  Picture Start Num: {self.pictureStartNum}")
        print(f"  Table Start Num: {self.tableStartNum}")
        print(f"  Equation Start Num: {self.equationStartNum}")
        print(f"  Caret Position: {self.caretPos}")

        # ID Mappings
        print(f"\n[ID Mappings]")
        print(f"  Binary Data Count: {self.binaryDataCount}")
        print(f"  Hangul Font Count: {self.hangulFontDataCount}")
        print(f"  English Font Count: {self.englishFontDataCount}")

        # Binary Items
        if self.binary_items:
            print(f"\n[Binary Data Items] ({len(self.binary_items)} items)")
            for item in self.binary_items:
                print(str(item))

        # Hangul Fonts
        if self.hangul_fonts:
            print(f"\n[Hangul Fonts] ({len(self.hangul_fonts)} fonts)")
            for font in self.hangul_fonts:
                print(str(font))

        # English Fonts
        if self.english_fonts:
            print(f"\n[English Fonts] ({len(self.english_fonts)} fonts)")
            for font in self.english_fonts:
                print(str(font))

        # Other Fonts
        if self.other_fonts:
            print(f"\n[Other Fonts] ({len(self.other_fonts)} fonts)")
            for font in self.other_fonts:
                print(str(font))

        # Character Properties
        if self.char_properties:
            print(f"\n[Character Properties] ({len(self.char_properties)} items)")
            for prop in self.char_properties:
                print(str(prop))

        # Styles
        if self.styles:
            print(f"\n[Styles] ({len(self.styles)} styles)")
            for style in self.styles:
                print(str(style))

        # Sections and Text - FULL
        for section in self.sections:
            print(f"\n[Section {section.index}]")
            print(f"  Paragraphs: {len(section.paragraphs)}")

            # All paragraphs - FULL
            if section.paragraphs:
                print(f"\n  [Paragraphs - Full Content]")
                for para in section.paragraphs:
                    print(f"  [{para.index}] (style={para.style_id})")
                    if para.text:
                        for line in para.text.split('\n'):
                            print(f"      {line}")

        # Full Text Content - NO TRUNCATION
        full_text = self.get_full_text()
        if full_text:
            print(f"\n[Full Text Content] ({len(full_text)} characters)")
            print("-" * 70)
            print(full_text)
            print("-" * 70)

        print(f"\n{'='*70}")


# Standalone helper functions
def remove_chinese_characters(s: str) -> str:
    """Remove Chinese characters from string"""
    return re.sub(r'[\u4e00-\u9fff]+', '', s)


def remove_control_characters(s: str) -> str:
    """Remove control characters from string"""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


# # Example usage
# if __name__ == "__main__":
#     import sys

#     verbose = "-v" in sys.argv or "--verbose" in sys.argv
#     args = [a for a in sys.argv[1:] if not a.startswith("-")]

#     if args:
#         file_path = args[0]
#         doc = Document.read_hwpx_document(file_path, verbose=verbose)
#         if doc:
#             if not verbose:
#                 doc.print_summary()

#             # Get cleaned text (without Chinese and control characters) - FULL
#             cleaned_text = doc.clean_text(remove_chinese=True, remove_control=True)
#             print(f"\n[Cleaned Text (no Chinese/control chars)] ({len(cleaned_text)} characters)")
#             print("-" * 70)
#             print(cleaned_text if cleaned_text else "(empty)")
#             print("-" * 70)

#             print("\n[Success] Document parsed successfully!")
#         else:
#             print("[Error] Failed to read HWPX document")
#     else:
#         print("Usage: python hwpx_extraction.py <file.hwpx> [-v|--verbose]")
#         print("\nOptions:")
#         print("  -v, --verbose  Show detailed parsing information during extraction")
#         print("\nOutput includes:")
#         print("  - Version info")
#         print("  - Archive file list")
#         print("  - Document properties (section count, start numbers)")
#         print("  - Binary data items (images, etc.)")
#         print("  - Font information (Hangul, English, other)")
#         print("  - Character properties and styles")
#         print("  - Parsed paragraphs with text content")
#         print("  - Full extracted text")
