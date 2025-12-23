import olefile
import zlib
import struct
import io
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import unicodedata
import re

# HWP Tag constants
HWPTAG_BEGIN = 0x010
HWPTAG_DOCUMENT_PROPERTIES = HWPTAG_BEGIN + 0
HWPTAG_ID_MAPPINGS = HWPTAG_BEGIN + 1
HWPTAG_BIN_DATA = HWPTAG_BEGIN + 2
HWPTAG_FACE_NAME = HWPTAG_BEGIN + 3
HWPTAG_BORDER_FILL = HWPTAG_BEGIN + 4
HWPTAG_CHAR_SHAPE = HWPTAG_BEGIN + 5
HWPTAG_TAB_DEF = HWPTAG_BEGIN + 6
HWPTAG_NUMBERING = HWPTAG_BEGIN + 7
HWPTAG_BULLET = HWPTAG_BEGIN + 8
HWPTAG_PARA_SHAPE = HWPTAG_BEGIN + 9
HWPTAG_STYLE = HWPTAG_BEGIN + 10
HWPTAG_DOC_DATA = HWPTAG_BEGIN + 11
HWPTAG_DISTRIBUTE_DOC_DATA = HWPTAG_BEGIN + 12
HWPTAG_COMPATIBLE_DOCUMENT = HWPTAG_BEGIN + 14
HWPTAG_LAYOUT_COMPATIBILITY = HWPTAG_BEGIN + 15

# Section tags
HWPTAG_PARA_HEADER = 0x042
HWPTAG_PARA_TEXT = 0x043
HWPTAG_PARA_CHAR_SHAPE = 0x044
HWPTAG_PARA_LINE_SEG = 0x045
HWPTAG_PARA_RANGE_TAG = 0x046
HWPTAG_CTRL_HEADER = 0x047
HWPTAG_LIST_HEADER = 0x048
HWPTAG_PAGE_DEF = 0x049
HWPTAG_FOOTNOTE_SHAPE = 0x04A
HWPTAG_PAGE_BORDER_FILL = 0x04B

# Tag name mapping for display
TAG_NAMES = {
    HWPTAG_DOCUMENT_PROPERTIES: "DOCUMENT_PROPERTIES",
    HWPTAG_ID_MAPPINGS: "ID_MAPPINGS",
    HWPTAG_BIN_DATA: "BIN_DATA",
    HWPTAG_FACE_NAME: "FACE_NAME",
    HWPTAG_BORDER_FILL: "BORDER_FILL",
    HWPTAG_CHAR_SHAPE: "CHAR_SHAPE",
    HWPTAG_TAB_DEF: "TAB_DEF",
    HWPTAG_NUMBERING: "NUMBERING",
    HWPTAG_BULLET: "BULLET",
    HWPTAG_PARA_SHAPE: "PARA_SHAPE",
    HWPTAG_STYLE: "STYLE",
    HWPTAG_DOC_DATA: "DOC_DATA",
    HWPTAG_DISTRIBUTE_DOC_DATA: "DISTRIBUTE_DOC_DATA",
    HWPTAG_COMPATIBLE_DOCUMENT: "COMPATIBLE_DOCUMENT",
    HWPTAG_LAYOUT_COMPATIBILITY: "LAYOUT_COMPATIBILITY",
    HWPTAG_PARA_HEADER: "PARA_HEADER",
    HWPTAG_PARA_TEXT: "PARA_TEXT",
    HWPTAG_PARA_CHAR_SHAPE: "PARA_CHAR_SHAPE",
    HWPTAG_PARA_LINE_SEG: "PARA_LINE_SEG",
    HWPTAG_PARA_RANGE_TAG: "PARA_RANGE_TAG",
    HWPTAG_CTRL_HEADER: "CTRL_HEADER",
    HWPTAG_LIST_HEADER: "LIST_HEADER",
    HWPTAG_PAGE_DEF: "PAGE_DEF",
    HWPTAG_FOOTNOTE_SHAPE: "FOOTNOTE_SHAPE",
    HWPTAG_PAGE_BORDER_FILL: "PAGE_BORDER_FILL",
}

# Special control characters in HWP text
CTRL_CHARS = {
    0: "NULL",
    1: "RESERVED1",
    2: "SECTION_DEF",       # 구역/단 정의
    3: "FIELD_START",       # 필드 시작
    4: "FIELD_END",         # 필드 끝
    5: "RESERVED5",
    6: "RESERVED6",
    7: "RESERVED7",
    8: "TITLE_MARK",        # 제목 표시
    9: "TAB",               # 탭
    10: "LINE_BREAK",       # 강제 줄 나눔
    11: "DRAWING_OBJ",      # 그리기 개체/표
    12: "RESERVED12",
    13: "PARA_BREAK",       # 문단 나눔
    14: "RESERVED14",
    15: "HIDDEN_DESC",      # 숨은 설명
    16: "HEADER_FOOTER",    # 머리말/꼬리말
    17: "FOOTNOTE",         # 각주/미주
    18: "AUTO_NUMBER",      # 자동 번호
    19: "RESERVED19",
    20: "RESERVED20",
    21: "PAGE_CTRL",        # 페이지 컨트롤
    22: "BOOKMARK",         # 책갈피/찾아보기 표시
    23: "CTRL_CHAR",        # 덧말/글자 겹침
    24: "HYPHEN",           # 하이픈
    25: "RESERVED25",
    26: "RESERVED26",
    27: "RESERVED27",
    28: "RESERVED28",
    29: "RESERVED29",
    30: "NONBREAK_SPACE",   # 묶음 빈칸
    31: "FIXWIDTH_SPACE",   # 고정폭 빈칸
}


def bytes_to_int(data: bytes, byteorder: str = 'little') -> int:
    """Convert bytes to integer"""
    return int.from_bytes(data, byteorder=byteorder)


def get_tag_name(tag_id: int) -> str:
    """Get human-readable tag name"""
    return TAG_NAMES.get(tag_id, f"UNKNOWN_0x{tag_id:03X}")


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
    flags: int = 0
    abs_path: str = ""
    rel_path: str = ""
    bin_id: int = 0
    extension: str = ""
    is_embedded: bool = False

    def __str__(self) -> str:
        if self.is_embedded:
            return f"  [{self.index}] Embedded: ID={self.bin_id}, Extension='{self.extension}'"
        else:
            return f"  [{self.index}] Linked: AbsPath='{self.abs_path}', RelPath='{self.rel_path}'"


@dataclass
class FontItem:
    index: int = 0
    name: str = ""
    alt_name: str = ""
    base_name: str = ""
    flags: int = 0

    def __str__(self) -> str:
        parts = [f"  [{self.index}] '{self.name}'"]
        if self.alt_name:
            parts.append(f"(alt: '{self.alt_name}')")
        if self.base_name:
            parts.append(f"(base: '{self.base_name}')")
        return " ".join(parts)


@dataclass
class Paragraph:
    index: int = 0
    text: str = ""
    char_count: int = 0
    control_mask: int = 0
    char_shape_count: int = 0
    range_tag_count: int = 0
    line_align_count: int = 0
    instance_id: int = 0

    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        preview = preview.replace('\n', '\\n').replace('\r', '\\r')
        return f"  [{self.index}] ({self.char_count} chars) \"{preview}\""


@dataclass
class RawRecord:
    """Raw record data for complete dump"""
    tag_id: int = 0
    tag_name: str = ""
    level: int = 0
    size: int = 0
    data: bytes = field(default_factory=bytes)
    offset: int = 0

    def __str__(self) -> str:
        hex_preview = self.data[:32].hex(' ') if self.data else "(empty)"
        if len(self.data) > 32:
            hex_preview += "..."
        return f"  [0x{self.offset:06X}] {self.tag_name:<25} Level={self.level}, Size={self.size:>5} | {hex_preview}"


@dataclass
class Record:
    tagID: int = 0
    level: int = 0
    size: int = 0
    stream: Optional[io.BytesIO] = None
    verbose: bool = False

    def read_record_header(self, tagID: int, stream: io.BytesIO) -> bool:
        """32비트 읽어 레코드 헤더 파싱"""
        self.stream = stream
        header_bytes = stream.read(4)
        if len(header_bytes) < 4:
            return False

        self.tagID, self.level, self.size = self.split_header_bits(header_bytes)

        if self.verbose:
            print(f"  [Record] TagID=0x{self.tagID:03X} ({get_tag_name(self.tagID)}), Level={self.level}, Size={self.size}")

        if self.tagID != tagID:  # 읽어온 레코드가 예상과 다른 레코드일 경우
            if self.verbose:
                print(f"  [Record] Expected TagID=0x{tagID:03X}, got 0x{self.tagID:03X}")
            return False

        # 확장 크기 처리
        if self.size == 0xFFF:  # 4095, 확장 크기
            size_bytes = stream.read(4)
            self.size = bytes_to_int(size_bytes)
            if self.verbose:
                print(f"  [Record] Extended size: {self.size}")

        return True

    def read_any_record(self, stream: io.BytesIO) -> bool:
        """Read any record without tag validation"""
        self.stream = stream
        header_bytes = stream.read(4)
        if len(header_bytes) < 4:
            return False

        self.tagID, self.level, self.size = self.split_header_bits(header_bytes)

        # 확장 크기 처리
        if self.size == 0xFFF:
            size_bytes = stream.read(4)
            self.size = bytes_to_int(size_bytes)

        return True

    def end_record(self) -> None:
        if self.size != 0:
            print(f'  [Warning] 레코드의 size만큼 데이터를 모두 읽어오지 않았습니다. (remaining: {self.size} bytes)')
            self.skip(self.size)

        self.tagID = 0
        self.level = 0
        self.size = 0

    def read_bytes(self, num_bytes: int) -> bytes:
        """지정 바이트 수만큼 데이터 읽기"""
        self.size -= num_bytes
        return self.stream.read(num_bytes)

    def skip(self, num_bytes: int) -> None:
        """데이터 건너뛰기"""
        self.size -= num_bytes
        self.stream.read(num_bytes)

    @staticmethod
    def split_header_bits(bits: bytes) -> tuple[int, int, int]:
        """리틀엔디안으로 읽어 32비트 헤더 분할 (10bit, 10bit, 12bit)"""
        num = struct.unpack('<I', bits)[0]
        return (
            (num >> 0) & 0x3FF,
            (num >> 10) & 0x3FF,
            (num >> 20) & 0xFFF
        )


@dataclass
class Section:
    index: int = 0
    paragraphs: List[Paragraph] = field(default_factory=list)
    raw_records: List[RawRecord] = field(default_factory=list)

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
    sections: List[Section] = field(default_factory=list)

    # Raw record dumps
    docinfo_records: List[RawRecord] = field(default_factory=list)

    # OLE stream info
    ole_streams: List[str] = field(default_factory=list)

    # File header info
    file_header: Dict[str, Any] = field(default_factory=dict)

    # Monitoring flag
    verbose: bool = False

    @classmethod
    def read_hwp_document(cls, file_path: str, verbose: bool = False) -> Optional['Document']:
        """HWP 파일 읽기"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Reading HWP file: {file_path}")
            print(f"{'='*70}")

        if not olefile.isOleFile(file_path):
            if verbose:
                print("[Error] Not a valid OLE file")
            return None

        ole = olefile.OleFileIO(file_path)
        doc = cls(verbose=verbose)

        # Store OLE stream list
        doc.ole_streams = ['/'.join(stream) for stream in ole.listdir()]

        if verbose:
            print("\n[OLE Streams]")
            for stream in doc.ole_streams:
                print(f"  - {stream}")

        # Read FileHeader
        doc._read_file_header(ole)

        # Read DocInfo
        if not doc._read_docinfo(ole):
            return None

        # Read BodyText sections
        doc._read_body_text(ole)

        ole.close()



        return doc

    def _read_file_header(self, ole: olefile.OleFileIO) -> bool:
        """Read FileHeader stream"""
        try:
            header_stream = ole.openstream('FileHeader')
            header_data = header_stream.read()

            if len(header_data) >= 256:
                signature = header_data[0:32]
                self.file_header['signature'] = signature.decode('utf-8', errors='replace').rstrip('\x00')
                self.file_header['version'] = f"{header_data[35]}.{header_data[34]}.{header_data[33]}.{header_data[32]}"

                properties = bytes_to_int(header_data[36:40])
                self.file_header['compressed'] = bool(properties & 0x01)
                self.file_header['encrypted'] = bool(properties & 0x02)
                self.file_header['distributed'] = bool(properties & 0x04)
                self.file_header['script'] = bool(properties & 0x08)
                self.file_header['drm'] = bool(properties & 0x10)
                self.file_header['xml_template'] = bool(properties & 0x20)
                self.file_header['history'] = bool(properties & 0x40)
                self.file_header['cert_signed'] = bool(properties & 0x80)
                self.file_header['cert_encrypted'] = bool(properties & 0x100)
                self.file_header['cert_drm'] = bool(properties & 0x200)
                self.file_header['ccl'] = bool(properties & 0x400)

                if self.verbose:
                    print(f"\n[FileHeader]")
                    print(f"  Signature: {self.file_header['signature']}")
                    print(f"  Version: {self.file_header['version']}")
                    print(f"  Compressed: {self.file_header['compressed']}")
                    print(f"  Encrypted: {self.file_header['encrypted']}")
                    print(f"  Distributed: {self.file_header['distributed']}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Error] Failed to read FileHeader: {e}")
            return False

    def _read_docinfo(self, ole: olefile.OleFileIO) -> bool:
        """Read and parse DocInfo stream"""
        try:
            doc_info_stream = ole.openstream('DocInfo')
            doc_info_data = doc_info_stream.read()

            if self.verbose:
                print(f"\n[DocInfo] Raw size: {len(doc_info_data)} bytes")

            # Decompress if needed
            if self.file_header.get('compressed', True):
                decompressed_data = zlib.decompress(doc_info_data, -15)
                if self.verbose:
                    print(f"[DocInfo] Decompressed size: {len(decompressed_data)} bytes")
            else:
                decompressed_data = doc_info_data

            stream = io.BytesIO(decompressed_data)

            # First pass: dump all raw records
            self._dump_all_records(stream, self.docinfo_records, "DocInfo")

            # Reset stream for structured parsing
            stream.seek(0)

            if not self.read_doc_info(stream):
                return False

            self.read_bin_data_item(stream)
            self.read_font_item(stream)

            return True
        except Exception as e:
            if self.verbose:
                print(f"[Error] Failed to read DocInfo: {e}")
            return False

    def _dump_all_records(self, stream: io.BytesIO, record_list: List[RawRecord], stream_name: str) -> None:
        """Dump all records from a stream"""
        if self.verbose:
            print(f"\n[Raw Records - {stream_name}]")

        start_pos = stream.tell()
        stream.seek(0, 2)  # Seek to end
        end_pos = stream.tell()
        stream.seek(start_pos)

        while stream.tell() < end_pos:
            offset = stream.tell()
            record = Record()
            if not record.read_any_record(stream):
                break

            data = stream.read(record.size)

            raw = RawRecord(
                tag_id=record.tagID,
                tag_name=get_tag_name(record.tagID),
                level=record.level,
                size=record.size,
                data=data,
                offset=offset
            )
            record_list.append(raw)

            if self.verbose:
                print(str(raw))

    def _read_body_text(self, ole: olefile.OleFileIO) -> None:
        """Read BodyText section streams"""
        section_idx = 0
        while True:
            stream_name = f'BodyText/Section{section_idx}'
            if not ole.exists(stream_name):
                break

            if self.verbose:
                print(f"\n[Reading {stream_name}]")

            try:
                section_stream = ole.openstream(stream_name)
                section_data = section_stream.read()

                if self.file_header.get('compressed', True):
                    section_data = zlib.decompress(section_data, -15)

                if self.verbose:
                    print(f"  Decompressed size: {len(section_data)} bytes")

                section = Section(index=section_idx)
                stream = io.BytesIO(section_data)

                # Dump raw records
                self._dump_all_records(stream, section.raw_records, stream_name)

                # Parse paragraphs
                stream.seek(0)
                self._parse_section_paragraphs(stream, section)

                self.sections.append(section)

            except Exception as e:
                if self.verbose:
                    print(f"  [Error] Failed to read section: {e}")

            section_idx += 1

        if self.verbose:
            print(f"\n[Total sections read: {len(self.sections)}]")

    def _parse_section_paragraphs(self, stream: io.BytesIO, section: Section) -> None:
        """Parse paragraphs from section stream"""
        stream.seek(0, 2)
        end_pos = stream.tell()
        stream.seek(0)

        para_idx = 0
        current_para = None

        while stream.tell() < end_pos:
            record = Record(verbose=False)
            if not record.read_any_record(stream):
                break

            if record.tagID == HWPTAG_PARA_HEADER:
                # Start of new paragraph
                if record.size >= 22:
                    data = stream.read(record.size)
                    current_para = Paragraph(index=para_idx)

                    # Parse paragraph header
                    current_para.char_count = bytes_to_int(data[0:4])
                    current_para.control_mask = bytes_to_int(data[4:8])
                    # para_shape_id = bytes_to_int(data[8:10])
                    # style_id = data[10]
                    # break_type = data[11]
                    current_para.char_shape_count = bytes_to_int(data[12:14])
                    current_para.range_tag_count = bytes_to_int(data[14:16])
                    current_para.line_align_count = bytes_to_int(data[16:18])
                    current_para.instance_id = bytes_to_int(data[18:22])

                    para_idx += 1
                else:
                    stream.read(record.size)

            elif record.tagID == HWPTAG_PARA_TEXT and current_para is not None:
                # Paragraph text content
                data = stream.read(record.size)
                current_para.text = self._decode_para_text(data)
                section.paragraphs.append(current_para)
                current_para = None

            else:
                # Skip other record types
                stream.read(record.size)

    def _decode_para_text(self, data: bytes) -> str:
        """Decode paragraph text data (UTF-16LE with control characters)"""
        result = []
        i = 0
        while i < len(data):
            if i + 1 >= len(data):
                break

            char_code = bytes_to_int(data[i:i+2])

            if char_code < 32:
                # Control character
                if char_code == 10:  # LINE_BREAK
                    result.append('\n')
                elif char_code == 13:  # PARA_BREAK
                    pass  # End of paragraph
                elif char_code == 9:  # TAB
                    result.append('\t')
                elif char_code == 30:  # NONBREAK_SPACE
                    result.append(' ')
                elif char_code == 31:  # FIXWIDTH_SPACE
                    result.append(' ')
                # Skip extended control characters (they use more bytes)
                if char_code in [2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23]:
                    # These use 8 extra bytes (inline extended)
                    i += 16
                    continue
                i += 2
            else:
                # Regular character
                try:
                    char = data[i:i+2].decode('utf-16-le')
                    result.append(char)
                except:
                    pass
                i += 2

        return ''.join(result)

    def read_doc_info(self, stream: io.BytesIO) -> bool:
        """문서 정보 읽기"""
        if self.verbose:
            print(f"\n[Parsing HWPTAG_DOCUMENT_PROPERTIES]")

        record = Record(verbose=self.verbose)
        if not record.read_record_header(HWPTAG_DOCUMENT_PROPERTIES, stream):
            return False

        self.sectionCount = bytes_to_int(record.read_bytes(2))
        self.pageStartNum = bytes_to_int(record.read_bytes(2))
        self.footnoteStartNum = bytes_to_int(record.read_bytes(2))
        self.endnoteStartNum = bytes_to_int(record.read_bytes(2))
        self.pictureStartNum = bytes_to_int(record.read_bytes(2))
        self.tableStartNum = bytes_to_int(record.read_bytes(2))
        self.equationStartNum = bytes_to_int(record.read_bytes(2))

        list_id = bytes_to_int(record.read_bytes(4))
        para_id = bytes_to_int(record.read_bytes(4))
        char_pos = bytes_to_int(record.read_bytes(4))
        self.caretPos = CaretPosition(list_id, para_id, char_pos)

        record.end_record()

        if self.verbose:
            print(f"\n[Document Properties]")
            print(f"  Section Count: {self.sectionCount}")
            print(f"  Page Start Num: {self.pageStartNum}")
            print(f"  Footnote Start Num: {self.footnoteStartNum}")
            print(f"  Endnote Start Num: {self.endnoteStartNum}")
            print(f"  Picture Start Num: {self.pictureStartNum}")
            print(f"  Table Start Num: {self.tableStartNum}")
            print(f"  Equation Start Num: {self.equationStartNum}")
            print(f"  Caret Position: {self.caretPos}")

        if not self.read_id_mapping(stream):
            return False

        return True

    def read_id_mapping(self, stream: io.BytesIO) -> bool:
        """ID 매핑 정보 읽기"""
        if self.verbose:
            print(f"\n[Parsing HWPTAG_ID_MAPPINGS]")

        record = Record(verbose=self.verbose)
        if not record.read_record_header(HWPTAG_ID_MAPPINGS, stream):
            return False

        self.binaryDataCount = bytes_to_int(record.read_bytes(4))
        self.hangulFontDataCount = bytes_to_int(record.read_bytes(4))
        self.englishFontDataCount = bytes_to_int(record.read_bytes(4))

        record.skip(4 * 5)  # Other font counts
        record.skip(4 * 10)  # Border/fill etc.

        record.end_record()

        if self.verbose:
            print(f"\n[ID Mappings]")
            print(f"  Binary Data Count: {self.binaryDataCount}")
            print(f"  Hangul Font Count: {self.hangulFontDataCount}")
            print(f"  English Font Count: {self.englishFontDataCount}")

        return True

    def read_bin_data_item(self, stream: io.BytesIO) -> bool:
        """바이너리 데이터 항목 읽기"""
        if self.binaryDataCount == 0:
            return True

        if self.verbose:
            print(f"\n[Parsing HWPTAG_BIN_DATA] ({self.binaryDataCount} items)")

        for i in range(self.binaryDataCount):
            record = Record(verbose=self.verbose)
            if not record.read_record_header(HWPTAG_BIN_DATA, stream):
                return False

            item = BinaryDataItem(index=i)
            item.flags = bytes_to_int(record.read_bytes(2))

            if (item.flags & 0x0000000F) == 0:
                item.is_embedded = False
                abs_path_len = bytes_to_int(record.read_bytes(2))
                item.abs_path = record.read_bytes(2 * abs_path_len).decode('utf-16-le')
                rel_path_len = bytes_to_int(record.read_bytes(2))
                item.rel_path = record.read_bytes(2 * rel_path_len).decode('utf-16-le')
            else:
                item.is_embedded = True
                item.bin_id = bytes_to_int(record.read_bytes(2))
                ext_len = bytes_to_int(record.read_bytes(2))
                item.extension = record.read_bytes(2 * ext_len).decode('utf-16-le')

            self.binary_items.append(item)
            record.end_record()

        if self.verbose:
            print(f"\n[Binary Data Items]")
            for item in self.binary_items:
                print(str(item))

        return True

    def read_font_item(self, stream: io.BytesIO) -> bool:
        """폰트 데이터 항목 읽기"""
        if self.hangulFontDataCount == 0:
            return True

        if self.verbose:
            print(f"\n[Parsing HWPTAG_FACE_NAME - Hangul Fonts] ({self.hangulFontDataCount} items)")

        for i in range(self.hangulFontDataCount):
            record = Record(verbose=self.verbose)
            if not record.read_record_header(HWPTAG_FACE_NAME, stream):
                return False

            item = FontItem(index=i)
            item.flags = bytes_to_int(record.read_bytes(1))
            font_len = bytes_to_int(record.read_bytes(2))
            item.name = record.read_bytes(2 * font_len).decode('utf-16-le')

            if item.flags & 0x80:
                _font_alt_flags = bytes_to_int(record.read_bytes(1))
                font_alt_len = bytes_to_int(record.read_bytes(2))
                item.alt_name = record.read_bytes(2 * font_alt_len).decode('utf-16-le')

            if item.flags & 0x40:
                record.skip(10)

            if item.flags & 0x20:
                base_len = bytes_to_int(record.read_bytes(2))
                item.base_name = record.read_bytes(2 * base_len).decode('utf-16-le')

            self.hangul_fonts.append(item)
            record.end_record()

        if self.verbose:
            print(f"\n[Hangul Fonts]")
            for font in self.hangul_fonts:
                print(str(font))

        return True

    def get_full_text(self) -> str:
        """Get all text from the document"""
        texts = []
        for section in self.sections:
            texts.append(section.get_full_text())
        return "\n\n".join(texts)

    def get_sections_data(self) -> List[Dict[str, Any]]:
        """Return all sections as list of structured dictionaries"""
        return [section.to_dict() for section in self.sections]

    def print_summary(self) -> None:
        """Print a summary of all extracted data"""
        print(f"\n{'='*70}")
        print("HWP DOCUMENT FULL EXTRACTION")
        print(f"{'='*70}")

        # File Header
        if self.file_header:
            print(f"\n[File Header]")
            print(f"  Signature: {self.file_header.get('signature', 'N/A')}")
            print(f"  Version: {self.file_header.get('version', 'N/A')}")
            print(f"  Compressed: {self.file_header.get('compressed', 'N/A')}")
            print(f"  Encrypted: {self.file_header.get('encrypted', 'N/A')}")

        # OLE Streams
        print(f"\n[OLE Streams] ({len(self.ole_streams)} streams)")
        for stream in self.ole_streams:
            print(f"  - {stream}")

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

        # Fonts
        if self.hangul_fonts:
            print(f"\n[Hangul Fonts] ({len(self.hangul_fonts)} fonts)")
            for font in self.hangul_fonts:
                print(str(font))

        # DocInfo Raw Records - FULL
        print(f"\n[DocInfo Records] ({len(self.docinfo_records)} records)")
        for rec in self.docinfo_records:
            print(str(rec))

        # Sections and Text - FULL
        for section in self.sections:
            print(f"\n[Section {section.index}]")
            print(f"  Raw Records: {len(section.raw_records)}")
            print(f"  Paragraphs: {len(section.paragraphs)}")

            # All raw records for this section
            if section.raw_records:
                print(f"\n  [Section {section.index} Raw Records]")
                for rec in section.raw_records:
                    print(f"  {rec}")

            # All paragraphs - FULL
            if section.paragraphs:
                print(f"\n  [Paragraphs - Full Content]")
                for para in section.paragraphs:
                    print(f"  [{para.index}] ({para.char_count} chars)")
                    if para.text:
                        # Print full text, indented
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


