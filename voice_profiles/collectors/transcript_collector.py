"""
Transcript Collector

Parses meeting transcripts (VTT, plain text, docx) to extract utterances by speaker.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import timedelta

from ..processors.disposition_extractor import TextSample


class TranscriptCollector:
    """
    Collects text samples from meeting transcripts.

    Supports:
    - VTT (WebVTT) files from Zoom, etc.
    - Plain text with speaker labels
    - Chunked JSON from CHORUS parsed data
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize collector.

        Args:
            data_path: Path to Data directory (default: auto-detect)
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Auto-detect from module location
            self.data_path = Path(__file__).parent.parent.parent / "Data"

        self.parsed_path = self.data_path / "parsed"

    def parse_vtt(self, vtt_path: str) -> List[Dict]:
        """
        Parse a VTT transcript file.

        Args:
            vtt_path: Path to .vtt file

        Returns:
            List of utterances with speaker, text, start_time, end_time
        """
        utterances = []
        path = Path(vtt_path)

        if not path.exists():
            return utterances

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # VTT format:
        # WEBVTT
        #
        # 00:00:00.000 --> 00:00:05.000
        # Speaker Name: Text here
        #
        # ...

        # Split into blocks
        blocks = re.split(r"\n\n+", content)

        for block in blocks:
            lines = block.strip().split("\n")

            # Skip header and empty blocks
            if not lines or lines[0] == "WEBVTT" or len(lines) < 2:
                continue

            # Look for timestamp line
            timestamp_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
                lines[0]
            )

            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                text_lines = lines[1:]
            else:
                # Try next line for timestamp
                if len(lines) > 1:
                    timestamp_match = re.match(
                        r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
                        lines[1]
                    )
                    if timestamp_match:
                        start_time = timestamp_match.group(1)
                        end_time = timestamp_match.group(2)
                        text_lines = lines[2:]
                    else:
                        continue
                else:
                    continue

            # Join text lines
            text = " ".join(text_lines).strip()

            if not text:
                continue

            # Extract speaker if present (format: "Speaker Name: text")
            speaker = None
            speaker_match = re.match(r"^([^:]+):\s*(.+)$", text)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()

            utterances.append({
                "speaker": speaker,
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
            })

        return utterances

    def parse_text_transcript(self, text_path: str) -> List[Dict]:
        """
        Parse a plain text transcript with speaker labels.

        Expects format like:
        Speaker Name: What they said
        Another Speaker: Their response

        Args:
            text_path: Path to text file

        Returns:
            List of utterances
        """
        utterances = []
        path = Path(text_path)

        if not path.exists():
            return utterances

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Pattern: Speaker at start of line followed by colon
        pattern = r"^([A-Z][^:]+):\s*(.+?)(?=\n[A-Z][^:]+:|$)"

        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            speaker = match.group(1).strip()
            text = match.group(2).strip()

            if text:
                utterances.append({
                    "speaker": speaker,
                    "text": text,
                    "start_time": None,
                    "end_time": None,
                })

        return utterances

    def get_chunks_by_source(self, source_pattern: str) -> List[Dict]:
        """
        Get parsed chunks matching a source pattern.

        Args:
            source_pattern: Pattern to match in source path (e.g., "APTO Retreat")

        Returns:
            List of chunk dicts with text and metadata
        """
        chunks_path = self.parsed_path / "chunks.json"

        if not chunks_path.exists():
            return []

        with open(chunks_path, "r") as f:
            all_chunks = json.load(f)

        matching = []
        for chunk in all_chunks:
            source = chunk.get("metadata", {}).get("source", "")
            if source_pattern.lower() in source.lower():
                matching.append(chunk)

        return matching

    def list_transcript_sources(self) -> List[Dict]:
        """
        List available transcript sources from parsed data.

        Returns:
            List of source info dicts
        """
        manifest_path = self.parsed_path / "manifest.json"

        if not manifest_path.exists():
            return []

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        transcript_keywords = ["transcript", "recording", ".vtt", "meeting"]

        sources = []
        for file_info in manifest.get("files", []):
            path = file_info.get("path", "")
            name = file_info.get("name", "")

            if any(kw in path.lower() or kw in name.lower() for kw in transcript_keywords):
                sources.append({
                    "name": name,
                    "path": path,
                    "num_elements": file_info.get("num_elements", 0),
                })

        return sources

    def collect_samples_by_speaker(
        self,
        speaker_name: str,
        sources: Optional[List[str]] = None,
        max_samples: int = 100,
        min_length: int = 30,
    ) -> List[TextSample]:
        """
        Collect text samples for a speaker from transcripts.

        Args:
            speaker_name: Name to match (partial match)
            sources: Specific source patterns to include
            max_samples: Maximum samples to collect
            min_length: Minimum utterance length

        Returns:
            List of TextSample objects
        """
        samples = []
        speaker_lower = speaker_name.lower()

        # Get transcript sources
        transcript_sources = self.list_transcript_sources()

        if sources:
            transcript_sources = [
                s for s in transcript_sources
                if any(src.lower() in s["path"].lower() for src in sources)
            ]

        # Process each source
        for source in transcript_sources:
            chunks = self.get_chunks_by_source(source["name"])

            for chunk in chunks:
                text = chunk.get("text", "")

                # Check if this chunk is from the target speaker
                # Chunks might have speaker info in metadata or text
                chunk_speaker = chunk.get("metadata", {}).get("speaker", "")

                # Also check if speaker name appears at start of text
                if speaker_lower in chunk_speaker.lower():
                    is_speaker = True
                elif text.lower().startswith(speaker_lower + ":"):
                    is_speaker = True
                    # Remove speaker prefix
                    text = text.split(":", 1)[1].strip() if ":" in text else text
                else:
                    is_speaker = False

                if not is_speaker:
                    continue

                if len(text) < min_length:
                    continue

                samples.append(TextSample(
                    text=text,
                    source_type="transcript",
                    source_id=f"transcript:{source['name']}",
                    timestamp=None,
                    context=source["name"],
                ))

                if len(samples) >= max_samples:
                    return samples

        return samples

    def collect_all_speakers(
        self,
        sources: Optional[List[str]] = None,
        min_utterances: int = 10,
        max_samples_per_speaker: int = 50,
    ) -> Dict[str, List[TextSample]]:
        """
        Collect samples for all speakers in transcripts.

        Args:
            sources: Specific source patterns to include
            min_utterances: Minimum utterances to include speaker
            max_samples_per_speaker: Max samples per speaker

        Returns:
            Dict mapping speaker_name -> list of TextSamples
        """
        # First pass: identify speakers and count utterances
        speaker_counts: Dict[str, int] = {}

        transcript_sources = self.list_transcript_sources()
        if sources:
            transcript_sources = [
                s for s in transcript_sources
                if any(src.lower() in s["path"].lower() for src in sources)
            ]

        for source in transcript_sources:
            # Try to find VTT files
            vtt_pattern = source["path"].replace(" ", "*")

            # For now, use chunks which may have speaker info
            chunks = self.get_chunks_by_source(source["name"])

            for chunk in chunks:
                speaker = chunk.get("metadata", {}).get("speaker", "")

                # Try to extract from text
                if not speaker:
                    text = chunk.get("text", "")
                    match = re.match(r"^([A-Z][^:]+):", text)
                    if match:
                        speaker = match.group(1).strip()

                if speaker and len(speaker) < 50:  # Sanity check
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        # Filter to speakers with enough utterances
        active_speakers = [
            name for name, count in speaker_counts.items()
            if count >= min_utterances
        ]

        # Collect samples
        result = {}
        for speaker in active_speakers:
            samples = self.collect_samples_by_speaker(
                speaker,
                sources=sources,
                max_samples=max_samples_per_speaker,
            )
            if samples:
                result[speaker] = samples

        return result

    def extract_questions_from_transcript(
        self,
        source_pattern: str,
        speaker_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract questions from a transcript.

        Args:
            source_pattern: Pattern to match source
            speaker_name: Optional filter to specific speaker

        Returns:
            List of question dicts with speaker, text, context
        """
        questions = []
        chunks = self.get_chunks_by_source(source_pattern)

        for chunk in chunks:
            text = chunk.get("text", "")
            speaker = chunk.get("metadata", {}).get("speaker", "")

            # Extract speaker from text if needed
            if not speaker:
                match = re.match(r"^([A-Z][^:]+):\s*(.+)$", text)
                if match:
                    speaker = match.group(1).strip()
                    text = match.group(2).strip()

            # Filter by speaker if specified
            if speaker_name and speaker_name.lower() not in speaker.lower():
                continue

            # Check if it's a question
            if "?" in text:
                questions.append({
                    "speaker": speaker,
                    "text": text,
                    "source": source_pattern,
                })

        return questions
