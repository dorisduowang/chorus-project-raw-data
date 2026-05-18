#!/usr/bin/env python3
"""
Advanced chunking strategies for different document types.

This module provides specialized chunking for:
- Transcripts (VTT, SRT, conversation formats) - preserves speaker turns and context
- Documents (PDF, DOCX) - respects section boundaries
- Structured data (CSV, JSON) - preserves record integrity
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import timedelta


@dataclass
class ChunkWithMetadata:
    """A text chunk with associated metadata for RAG retrieval."""
    text: str
    source_file: str = ""
    source_path: str = ""
    element_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    # Enhanced metadata
    doc_type: str = ""  # proposal, transcript, cv, newsletter, meeting, report, paper, other
    date_year: Optional[int] = None
    date_month: Optional[int] = None
    # Transcript-specific metadata
    timestamp_start: Optional[str] = None  # "00:05:30"
    timestamp_end: Optional[str] = None
    speakers: Optional[str] = None  # Comma-separated speaker names
    # Rich extraction metadata (populated by enrich_chunk)
    entities_people: Optional[List[str]] = None
    entities_orgs: Optional[List[str]] = None
    entities_projects: Optional[List[str]] = None
    entities_grants: Optional[List[str]] = None
    dates_mentioned: Optional[List[str]] = None  # ISO format dates found in text
    topics: Optional[List[str]] = None  # Topic classifications
    key_terms: Optional[List[str]] = None  # Important terms

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'source_file': self.source_file,
            'source_path': self.source_path,
            'element_type': self.element_type,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'page_number': self.page_number,
            'section': self.section,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'doc_type': self.doc_type,
            'date_year': self.date_year,
            'date_month': self.date_month,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'speakers': self.speakers,
            'entities_people': self.entities_people,
            'entities_orgs': self.entities_orgs,
            'entities_projects': self.entities_projects,
            'entities_grants': self.entities_grants,
            'dates_mentioned': self.dates_mentioned,
            'topics': self.topics,
            'key_terms': self.key_terms,
        }

    def citation(self) -> str:
        parts = [self.source_file] if self.source_file else []
        if self.timestamp_start:
            parts.append(f"@{self.timestamp_start}")
        elif self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.section:
            parts.append(f"§{self.section}")
        return " | ".join(parts) if parts else "Unknown"


# =============================================================================
# VTT/SRT Transcript Parsing
# =============================================================================

@dataclass
class TranscriptSegment:
    """A segment from a transcript with timing and speaker info."""
    text: str
    start_time: str  # "00:05:30.000"
    end_time: str
    speaker: Optional[str] = None

    def start_seconds(self) -> float:
        """Convert start_time to seconds."""
        return parse_vtt_timestamp(self.start_time)

    def end_seconds(self) -> float:
        """Convert end_time to seconds."""
        return parse_vtt_timestamp(self.end_time)


def parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT timestamp to seconds. Handles both HH:MM:SS.mmm and MM:SS.mmm"""
    parts = ts.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return 0.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS for citations."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_vtt_content(content: str) -> List[TranscriptSegment]:
    """
    Parse VTT content into transcript segments.

    VTT format:
    WEBVTT

    00:00:00.000 --> 00:00:03.500
    Speaker Name: Hello everyone...

    00:00:03.500 --> 00:00:07.000
    Text continues...
    """
    segments = []

    # Pattern for VTT timestamps
    timestamp_pattern = r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})'

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip WEBVTT header and empty lines
        if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
            i += 1
            continue

        # Look for timestamp line
        match = re.match(timestamp_pattern, line)
        if match:
            start_time = match.group(1)
            end_time = match.group(2)

            # Collect text lines until next timestamp or empty line
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(timestamp_pattern, lines[i]):
                text_lines.append(lines[i].strip())
                i += 1

            text = ' '.join(text_lines)

            # Try to extract speaker from text (common formats)
            speaker = None
            # Format: "Speaker Name: text"
            speaker_match = re.match(r'^([A-Za-z][A-Za-z\s\.]+):\s*(.+)$', text)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()

            if text:
                segments.append(TranscriptSegment(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker
                ))
        else:
            i += 1

    return segments


def chunk_transcript_by_time(
    segments: List[TranscriptSegment],
    target_duration_seconds: float = 180,  # 3 minutes per chunk
    min_duration_seconds: float = 60,      # At least 1 minute
    max_duration_seconds: float = 300,     # At most 5 minutes
    overlap_seconds: float = 30            # 30 second overlap
) -> List[Tuple[str, str, str, str]]:
    """
    Chunk transcript segments by time, preserving conversational context.

    Returns:
        List of tuples: (chunk_text, start_timestamp, end_timestamp, speakers)
    """
    if not segments:
        return []

    chunks = []
    current_segments = []
    current_speakers = set()
    current_start = segments[0].start_seconds()

    for segment in segments:
        current_segments.append(segment)
        if segment.speaker:
            current_speakers.add(segment.speaker)

        current_duration = segment.end_seconds() - current_start

        # Check if we should close this chunk
        if current_duration >= target_duration_seconds:
            # Build chunk text with speaker attribution
            chunk_text = build_transcript_chunk_text(current_segments)
            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(segment.end_seconds())
            speakers = ", ".join(sorted(current_speakers)) if current_speakers else ""

            chunks.append((chunk_text, start_ts, end_ts, speakers))

            # Find overlap point - go back ~overlap_seconds
            overlap_target = segment.end_seconds() - overlap_seconds
            overlap_segments = []
            for s in current_segments:
                if s.start_seconds() >= overlap_target:
                    overlap_segments.append(s)

            # Start new chunk with overlap
            current_segments = overlap_segments
            current_speakers = {s.speaker for s in overlap_segments if s.speaker}
            if overlap_segments:
                current_start = overlap_segments[0].start_seconds()
            else:
                current_start = segment.end_seconds()

    # Handle remaining segments
    if current_segments:
        chunk_text = build_transcript_chunk_text(current_segments)
        start_ts = format_timestamp(current_start)
        end_ts = format_timestamp(current_segments[-1].end_seconds())
        speakers = ", ".join(sorted(current_speakers)) if current_speakers else ""
        chunks.append((chunk_text, start_ts, end_ts, speakers))

    return chunks


def build_transcript_chunk_text(segments: List[TranscriptSegment]) -> str:
    """
    Build readable chunk text from transcript segments.
    Groups consecutive segments by speaker for readability.
    """
    if not segments:
        return ""

    lines = []
    current_speaker = None
    current_text_parts = []

    for segment in segments:
        if segment.speaker != current_speaker:
            # Flush previous speaker's text
            if current_text_parts:
                if current_speaker:
                    lines.append(f"{current_speaker}: {' '.join(current_text_parts)}")
                else:
                    lines.append(' '.join(current_text_parts))
            current_speaker = segment.speaker
            current_text_parts = [segment.text]
        else:
            current_text_parts.append(segment.text)

    # Flush final speaker's text
    if current_text_parts:
        if current_speaker:
            lines.append(f"{current_speaker}: {' '.join(current_text_parts)}")
        else:
            lines.append(' '.join(current_text_parts))

    return '\n\n'.join(lines)


# =============================================================================
# Generic Text Chunking (for non-transcripts)
# =============================================================================

def chunk_text_with_overlap(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk_size: int = 50
) -> List[Tuple[str, int, int]]:
    """
    Split text into chunks with overlap, respecting sentence boundaries.

    Returns:
        List of tuples: (chunk_text, char_start, char_end)
    """
    if not text or len(text) < min_chunk_size:
        return [(text, 0, len(text))] if text else []

    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if len(current_chunk) + sentence_len > chunk_size and current_chunk:
            chunk_end = char_pos
            chunks.append((current_chunk.strip(), current_start, chunk_end))

            if overlap > 0 and len(current_chunk) > overlap:
                overlap_start = len(current_chunk) - overlap
                overlap_text = current_chunk[overlap_start:]
                current_chunk = overlap_text + " " + sentence
                current_start = chunk_end - len(overlap_text)
            else:
                current_chunk = sentence
                current_start = char_pos
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                current_start = char_pos

        char_pos += sentence_len + 1

    if current_chunk.strip():
        if len(current_chunk.strip()) >= min_chunk_size or not chunks:
            chunks.append((current_chunk.strip(), current_start, len(text)))
        elif chunks:
            prev_text, prev_start, _ = chunks[-1]
            chunks[-1] = (prev_text + " " + current_chunk.strip(), prev_start, len(text))

    return chunks


# =============================================================================
# Document Type Detection
# =============================================================================

def detect_doc_type(filename: str) -> str:
    """Detect document type from filename."""
    filename_lower = filename.lower()

    if "transcript" in filename_lower or filename_lower.endswith(('.vtt', '.srt')):
        return "transcript"
    elif any(x in filename_lower for x in ["proposal", "grant", "muri", "submission"]):
        return "proposal"
    elif any(x in filename_lower for x in ["cv", "biosketch", "resume"]):
        return "cv"
    elif "newsletter" in filename_lower:
        return "newsletter"
    elif any(x in filename_lower for x in ["meeting", "weekly", "agenda"]):
        return "meeting"
    elif "report" in filename_lower:
        return "report"
    elif any(x in filename_lower for x in ["recording", "zoom", "gmt"]):
        return "transcript"  # Zoom recordings often have "GMT" prefix
    elif filename_lower.endswith('.pdf') and "paper" in filename_lower:
        return "paper"
    else:
        return "other"


def is_parsed_vtt_content(content: str) -> bool:
    """
    Detect if content is pre-parsed VTT (e.g., from Unstructured parsing).

    Pre-parsed VTT has timestamp lines embedded in the text like:
    "text here\n\n123\n\n00:05:30.000 --> 00:05:35.000\n\nMore text"
    """
    # Look for VTT timestamp patterns scattered in content
    timestamp_pattern = r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}'
    matches = re.findall(timestamp_pattern, content)
    # If we find multiple timestamp patterns, it's likely parsed VTT
    return len(matches) >= 5


def parse_embedded_vtt_content(content: str) -> List[TranscriptSegment]:
    """
    Parse content that has embedded VTT timestamps from pre-processing.

    Handles two formats from Unstructured:
    1. Separate elements: [text, cue_num, timestamp, text, ...]
    2. Combined elements: "cue_num timestamp --> timestamp text"
    """
    segments = []

    # Split on double newlines to get individual elements
    parts = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Pattern for standalone timestamp
    standalone_ts = r'^(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*(?:->|-->)\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})$'

    # Pattern for combined: "cue_num timestamp --> timestamp text"
    combined_pattern = r'^(\d+)\s+(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*(?:->|-->)\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\s+(.+)$'

    i = 0
    while i < len(parts):
        part = parts[i]

        # Try combined format first (more common in the data)
        combined_match = re.match(combined_pattern, part, re.DOTALL)
        if combined_match:
            cue_num, start_time, end_time, text = combined_match.groups()
            text = text.strip()

            if text:
                # Extract speaker
                speaker = None
                speaker_match = re.match(r'^([A-Za-z][A-Za-z\s\.\-]+):\s*(.+)$', text, re.DOTALL)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    text = speaker_match.group(2).strip()

                segments.append(TranscriptSegment(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker
                ))
            i += 1
            continue

        # Try standalone timestamp format
        standalone_match = re.match(standalone_ts, part)
        if standalone_match:
            start_time = standalone_match.group(1)
            end_time = standalone_match.group(2)

            # Look for text after this timestamp (skip any cue numbers)
            i += 1
            text = None
            while i < len(parts):
                next_part = parts[i]
                # Skip cue numbers
                if next_part.isdigit():
                    i += 1
                    continue
                # Stop if we hit another timestamp
                if re.match(standalone_ts, next_part) or re.match(combined_pattern, next_part):
                    break
                # This is text
                text = next_part
                i += 1
                break

            if text:
                # Extract speaker
                speaker = None
                speaker_match = re.match(r'^([A-Za-z][A-Za-z\s\.\-]+):\s*(.+)$', text, re.DOTALL)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    text = speaker_match.group(2).strip()

                segments.append(TranscriptSegment(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker
                ))
            continue

        # Not a timestamp - skip
        i += 1

    return segments


def extract_date_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract year and month from filename patterns."""
    year = None
    month = None

    # Look for 4-digit years (2020-2029)
    year_match = re.search(r'(20[12]\d)', filename)
    if year_match:
        year = int(year_match.group(1))

    # Look for FY patterns (FY24, FY25)
    fy_match = re.search(r'FY(\d{2})', filename, re.IGNORECASE)
    if fy_match:
        fy_year = int(fy_match.group(1))
        year = 2000 + fy_year if fy_year < 50 else 1900 + fy_year

    # Look for month patterns
    month_names = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }

    filename_lower = filename.lower()
    for month_name, month_num in month_names.items():
        if month_name in filename_lower:
            month = month_num
            break

    # Look for numeric month patterns (01-12)
    month_match = re.search(r'[-_](\d{2})[-_]', filename)
    if month_match:
        potential_month = int(month_match.group(1))
        if 1 <= potential_month <= 12:
            month = potential_month

    return year, month


# =============================================================================
# Main Chunking Function
# =============================================================================

def smart_chunk_document(
    content: str,
    filename: str,
    filepath: str = "",
    chunk_size: int = 500,
    overlap: int = 100,
    transcript_duration: float = 180  # 3 minutes for transcripts
) -> List[ChunkWithMetadata]:
    """
    Intelligently chunk a document based on its type.

    Returns:
        List of ChunkWithMetadata objects
    """
    doc_type = detect_doc_type(filename)
    date_year, date_month = extract_date_from_filename(filename)

    chunks = []

    # Handle transcripts specially - check for VTT content in multiple ways
    is_vtt = (
        filename.lower().endswith('.vtt') or
        'WEBVTT' in content[:100] or
        (doc_type == "transcript" and is_parsed_vtt_content(content))
    )

    if is_vtt:
        # Try parsing as standard VTT first
        segments = parse_vtt_content(content)

        # If standard parsing found few segments, try embedded format
        if len(segments) < 5 and is_parsed_vtt_content(content):
            segments = parse_embedded_vtt_content(content)

        if segments:
            time_chunks = chunk_transcript_by_time(
                segments,
                target_duration_seconds=transcript_duration,
                overlap_seconds=30
            )

            for i, (text, start_ts, end_ts, speakers) in enumerate(time_chunks):
                chunk = ChunkWithMetadata(
                    text=text,
                    source_file=filename,
                    source_path=filepath,
                    element_type="Transcript",
                    chunk_index=i,
                    total_chunks=len(time_chunks),
                    doc_type=doc_type,
                    date_year=date_year,
                    date_month=date_month,
                    timestamp_start=start_ts,
                    timestamp_end=end_ts,
                    speakers=speakers
                )
                chunks.append(chunk)

            return chunks

    # Default text chunking for non-transcripts
    text_chunks = chunk_text_with_overlap(content, chunk_size, overlap)

    for i, (text, char_start, char_end) in enumerate(text_chunks):
        chunk = ChunkWithMetadata(
            text=text,
            source_file=filename,
            source_path=filepath,
            element_type="Text",
            chunk_index=i,
            total_chunks=len(text_chunks),
            char_start=char_start,
            char_end=char_end,
            doc_type=doc_type,
            date_year=date_year,
            date_month=date_month
        )
        chunks.append(chunk)

    return chunks


# =============================================================================
# Batch Processing
# =============================================================================

def process_file_list(
    files: List[dict],  # List of {"name": str, "path": str, "content": str}
    chunk_size: int = 500,
    overlap: int = 100,
    transcript_duration: float = 180
) -> List[ChunkWithMetadata]:
    """
    Process a list of files with smart chunking.

    Args:
        files: List of dicts with 'name', 'path', and 'content' keys
        chunk_size: Default chunk size for text documents
        overlap: Overlap between chunks
        transcript_duration: Target duration in seconds for transcript chunks

    Returns:
        List of all chunks from all files
    """
    all_chunks = []

    for file_info in files:
        name = file_info.get('name', 'Unknown')
        path = file_info.get('path', name)
        content = file_info.get('content', '')

        if not content:
            continue

        file_chunks = smart_chunk_document(
            content=content,
            filename=name,
            filepath=path,
            chunk_size=chunk_size,
            overlap=overlap,
            transcript_duration=transcript_duration
        )

        all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    # Test with sample VTT content
    sample_vtt = """WEBVTT

00:00:00.000 --> 00:00:05.000
James Evans: Welcome everyone to the weekly lab meeting.

00:00:05.000 --> 00:00:12.000
James Evans: Today we're going to discuss the APTO project updates and some new research directions.

00:00:12.000 --> 00:00:20.000
Student A: Thanks James. I've been working on the network analysis component.

00:00:20.000 --> 00:00:35.000
Student A: We found some interesting patterns in the citation networks that suggest emerging fields.

00:00:35.000 --> 00:00:45.000
James Evans: That's exciting. Can you share your methodology?

00:00:45.000 --> 00:01:00.000
Student A: Sure, we're using a combination of temporal embeddings and community detection.
"""

    print("Testing VTT parsing...")
    segments = parse_vtt_content(sample_vtt)
    print(f"Parsed {len(segments)} segments")
    for s in segments:
        print(f"  [{s.start_time}] {s.speaker or 'Unknown'}: {s.text[:50]}...")

    print("\nTesting smart chunking...")
    chunks = smart_chunk_document(sample_vtt, "test_recording.vtt")
    print(f"Created {len(chunks)} chunks")
    for c in chunks:
        print(f"  [{c.timestamp_start}] {c.speakers}: {c.text[:80]}...")
