#!/usr/bin/env python3
"""
Content-Based Date Extraction Module for RAG System

Extracts dates mentioned within document text content using regex patterns
and dateutil parsing. Designed for fast processing in chunking pipelines.

Usage:
    from extractors import extract_dates, get_date_extractor

    result = extract_dates("The deadline is March 15, 2024")
    print(result.primary_date.normalized)  # "2024-03-15"
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

try:
    from dateutil import parser as dateutil_parser
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False


@dataclass
class ExtractedDate:
    """A single date extracted from text."""
    raw_text: str                    # Original text matched ("March 15, 2024")
    normalized: str                  # ISO format ("2024-03-15" or "2024-03")
    year: int                        # 2024
    month: Optional[int]             # 3 (or None for partial dates)
    day: Optional[int]               # 15 (or None for partial dates)
    context: str                     # "deadline", "event", "publication", "project", "unknown"
    confidence: float                # 0.0-1.0
    char_start: int = 0              # Position in original text
    char_end: int = 0                # Position in original text
    is_range: bool = False           # True if this is a date range
    range_end: Optional[str] = None  # End date for ranges ("2024-03-20")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_text": self.raw_text,
            "normalized": self.normalized,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "context": self.context,
            "confidence": self.confidence,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "is_range": self.is_range,
            "range_end": self.range_end,
        }

    def is_partial(self) -> bool:
        """Check if this is a partial date (year-month or year only)."""
        return self.day is None


@dataclass
class DateExtractionResult:
    """Result of date extraction from text."""
    dates: List[ExtractedDate] = field(default_factory=list)
    primary_date: Optional[ExtractedDate] = None  # Most likely relevant date

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dates": [d.to_dict() for d in self.dates],
            "primary_date": self.primary_date.to_dict() if self.primary_date else None,
        }

    def has_dates(self) -> bool:
        """Check if any dates were found."""
        return len(self.dates) > 0

    def get_by_context(self, context: str) -> List[ExtractedDate]:
        """Get all dates with a specific context."""
        return [d for d in self.dates if d.context == context]


class DateExtractor:
    """
    Fast content-based date extractor using regex + dateutil.

    Extracts various date formats and attempts to identify their context
    (deadline, event, publication, etc.) from surrounding text.
    """

    # Month name pattern
    MONTH_NAMES = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'

    def __init__(self):
        """Initialize date extractor with compiled regex patterns."""
        self._compile_patterns()

        # Context detection keywords
        self.context_patterns = {
            "deadline": re.compile(r'\b(deadline|due|submit|submission|cutoff)\b', re.IGNORECASE),
            "event": re.compile(r'\b(meeting|conference|workshop|seminar|talk|presentation|symposium)\b', re.IGNORECASE),
            "publication": re.compile(r'\b(published|released|issued|appeared|print)\b', re.IGNORECASE),
            "project": re.compile(r'\b(started|began|ends|ended|completed|finish|launch|timeline)\b', re.IGNORECASE),
        }

    def _compile_patterns(self):
        """Compile all date regex patterns."""

        # ISO dates (2024-03-15, 2024-03)
        self.iso_date = re.compile(
            r'\b(20[12]\d)[-/](0[1-9]|1[0-2])(?:[-/](0[1-9]|[12]\d|3[01]))?\b'
        )

        # US format (03/15/2024, 3/15/24)
        self.us_date = re.compile(
            r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{2}|\d{4})\b'
        )

        # Text dates (March 15, 2024 | Mar 15, 2024)
        self.text_date_mdy = re.compile(
            rf'\b({self.MONTH_NAMES})\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Text dates (15 March 2024)
        self.text_date_dmy = re.compile(
            rf'\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+({self.MONTH_NAMES}),?\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Partial dates (March 2024, Mar 2024)
        self.text_month_year = re.compile(
            rf'\b({self.MONTH_NAMES})\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Quarter dates (Q1 2024, Q1-2024)
        self.quarter_date = re.compile(
            r'\bQ([1-4])[-\s]*(20[12]\d)\b',
            re.IGNORECASE
        )

        # Fiscal year (FY24, FY2024)
        self.fiscal_year = re.compile(
            r'\bFY[-\s]?(\d{2}|\d{4})\b',
            re.IGNORECASE
        )

        # Seasonal (Spring 2024, Summer 2024)
        self.seasonal = re.compile(
            r'\b(Spring|Summer|Fall|Autumn|Winter)\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Year range (2024-2025)
        self.year_range = re.compile(
            r'\b(20[12]\d)[-–](20[12]\d)\b'
        )

        # Date range (March 15-20, 2024)
        self.date_range_text = re.compile(
            rf'\b({self.MONTH_NAMES})\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[-–]+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Year with context (in 2023, since 2024)
        self.year_context = re.compile(
            r'\b(?:in|since|during|by|year|dated|from|until|through)\s+(20[12]\d)\b',
            re.IGNORECASE
        )

        # Month name to number mapping
        self.month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
            'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }

    def extract(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> DateExtractionResult:
        """
        Extract dates from text content.

        Args:
            text: Text to extract dates from
            reference_date: Reference date for resolving relative dates (default: today)

        Returns:
            DateExtractionResult with all found dates
        """
        if not text:
            return DateExtractionResult()

        if reference_date is None:
            reference_date = datetime.now()

        all_matches: List[ExtractedDate] = []

        # Extract each pattern type
        all_matches.extend(self._extract_iso_dates(text))
        all_matches.extend(self._extract_us_dates(text))
        all_matches.extend(self._extract_text_dates(text))
        all_matches.extend(self._extract_partial_dates(text))
        all_matches.extend(self._extract_quarter_dates(text))
        all_matches.extend(self._extract_fiscal_years(text))
        all_matches.extend(self._extract_seasonal_dates(text))
        all_matches.extend(self._extract_date_ranges(text))
        all_matches.extend(self._extract_year_context(text))

        # Deduplicate overlapping matches
        deduplicated = self._deduplicate_dates(all_matches)

        # Determine context for each date
        for date in deduplicated:
            context, confidence_boost = self._determine_context(
                text, date.char_start, date.char_end
            )
            date.context = context
            date.confidence = min(1.0, date.confidence + confidence_boost)

        # Select primary date
        primary = self._select_primary_date(deduplicated)

        return DateExtractionResult(dates=deduplicated, primary_date=primary)

    def _extract_iso_dates(self, text: str) -> List[ExtractedDate]:
        """Extract ISO format dates (2024-03-15, 2024-03)."""
        dates = []
        for match in self.iso_date.finditer(text):
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3)) if match.group(3) else None

            if day:
                normalized = f"{year:04d}-{month:02d}-{day:02d}"
            else:
                normalized = f"{year:04d}-{month:02d}"

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=normalized,
                year=year,
                month=month,
                day=day,
                context="unknown",
                confidence=0.95,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _extract_us_dates(self, text: str) -> List[ExtractedDate]:
        """Extract US format dates (03/15/2024, 3/15/24)."""
        dates = []
        for match in self.us_date.finditer(text):
            try:
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))

                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year

                # Validate
                datetime(year, month, day)

                normalized = f"{year:04d}-{month:02d}-{day:02d}"

                dates.append(ExtractedDate(
                    raw_text=match.group(0),
                    normalized=normalized,
                    year=year,
                    month=month,
                    day=day,
                    context="unknown",
                    confidence=0.85,
                    char_start=match.start(),
                    char_end=match.end()
                ))
            except (ValueError, OverflowError):
                continue
        return dates

    def _extract_text_dates(self, text: str) -> List[ExtractedDate]:
        """Extract text format dates (March 15, 2024)."""
        dates = []

        # Month-Day-Year
        for match in self.text_date_mdy.finditer(text):
            try:
                month_str = match.group(1).lower()[:3]
                month = self.month_map.get(month_str, 1)
                day = int(match.group(2))
                year = int(match.group(3))

                datetime(year, month, day)
                normalized = f"{year:04d}-{month:02d}-{day:02d}"

                dates.append(ExtractedDate(
                    raw_text=match.group(0),
                    normalized=normalized,
                    year=year,
                    month=month,
                    day=day,
                    context="unknown",
                    confidence=0.9,
                    char_start=match.start(),
                    char_end=match.end()
                ))
            except (ValueError, OverflowError):
                continue

        # Day-Month-Year
        for match in self.text_date_dmy.finditer(text):
            try:
                day = int(match.group(1))
                month_str = match.group(2).lower()[:3]
                month = self.month_map.get(month_str, 1)
                year = int(match.group(3))

                datetime(year, month, day)
                normalized = f"{year:04d}-{month:02d}-{day:02d}"

                dates.append(ExtractedDate(
                    raw_text=match.group(0),
                    normalized=normalized,
                    year=year,
                    month=month,
                    day=day,
                    context="unknown",
                    confidence=0.9,
                    char_start=match.start(),
                    char_end=match.end()
                ))
            except (ValueError, OverflowError):
                continue

        return dates

    def _extract_partial_dates(self, text: str) -> List[ExtractedDate]:
        """Extract partial dates (March 2024)."""
        dates = []
        for match in self.text_month_year.finditer(text):
            month_str = match.group(1).lower()[:3]
            month = self.month_map.get(month_str, 1)
            year = int(match.group(2))

            normalized = f"{year:04d}-{month:02d}"

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=normalized,
                year=year,
                month=month,
                day=None,
                context="unknown",
                confidence=0.8,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _extract_quarter_dates(self, text: str) -> List[ExtractedDate]:
        """Extract quarter dates (Q1 2024)."""
        dates = []
        for match in self.quarter_date.finditer(text):
            quarter = int(match.group(1))
            year = int(match.group(2))
            month = (quarter - 1) * 3 + 1

            normalized = f"{year:04d}-{month:02d}"

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=normalized,
                year=year,
                month=month,
                day=None,
                context="unknown",
                confidence=0.85,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _extract_fiscal_years(self, text: str) -> List[ExtractedDate]:
        """Extract fiscal years (FY24, FY2024)."""
        dates = []
        for match in self.fiscal_year.finditer(text):
            year = int(match.group(1))
            if year < 100:
                year = 2000 + year if year < 50 else 1900 + year

            normalized = f"{year:04d}"

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=normalized,
                year=year,
                month=None,
                day=None,
                context="unknown",
                confidence=0.75,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _extract_seasonal_dates(self, text: str) -> List[ExtractedDate]:
        """Extract seasonal dates (Spring 2024)."""
        dates = []
        season_months = {
            "spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12,
        }

        for match in self.seasonal.finditer(text):
            season = match.group(1).lower()
            year = int(match.group(2))
            month = season_months.get(season, 1)

            normalized = f"{year:04d}-{month:02d}"

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=normalized,
                year=year,
                month=month,
                day=None,
                context="unknown",
                confidence=0.7,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _extract_date_ranges(self, text: str) -> List[ExtractedDate]:
        """Extract date ranges."""
        dates = []

        # Year ranges (2024-2025)
        for match in self.year_range.finditer(text):
            start_year = int(match.group(1))
            end_year = int(match.group(2))

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=f"{start_year:04d}",
                year=start_year,
                month=None,
                day=None,
                context="project",
                confidence=0.75,
                char_start=match.start(),
                char_end=match.end(),
                is_range=True,
                range_end=f"{end_year:04d}"
            ))

        # Date ranges (March 15-20, 2024)
        for match in self.date_range_text.finditer(text):
            try:
                month_str = match.group(1).lower()[:3]
                month = self.month_map.get(month_str, 1)
                start_day = int(match.group(2))
                end_day = int(match.group(3))
                year = int(match.group(4))

                datetime(year, month, start_day)
                datetime(year, month, end_day)

                normalized = f"{year:04d}-{month:02d}-{start_day:02d}"
                range_end = f"{year:04d}-{month:02d}-{end_day:02d}"

                dates.append(ExtractedDate(
                    raw_text=match.group(0),
                    normalized=normalized,
                    year=year,
                    month=month,
                    day=start_day,
                    context="event",
                    confidence=0.85,
                    char_start=match.start(),
                    char_end=match.end(),
                    is_range=True,
                    range_end=range_end
                ))
            except (ValueError, OverflowError):
                continue

        return dates

    def _extract_year_context(self, text: str) -> List[ExtractedDate]:
        """Extract years with context words."""
        dates = []
        for match in self.year_context.finditer(text):
            year = int(match.group(1))

            dates.append(ExtractedDate(
                raw_text=match.group(0),
                normalized=f"{year:04d}",
                year=year,
                month=None,
                day=None,
                context="unknown",
                confidence=0.5,
                char_start=match.start(),
                char_end=match.end()
            ))
        return dates

    def _determine_context(
        self,
        text: str,
        char_start: int,
        char_end: int
    ) -> Tuple[str, float]:
        """Determine date context from surrounding text."""
        window_start = max(0, char_start - 50)
        window_end = min(len(text), char_end + 50)
        context_window = text[window_start:window_end].lower()

        for ctx_type, pattern in self.context_patterns.items():
            if pattern.search(context_window):
                return ctx_type, 0.1

        return "unknown", 0.0

    def _deduplicate_dates(self, dates: List[ExtractedDate]) -> List[ExtractedDate]:
        """Remove overlapping date matches."""
        if not dates:
            return []

        sorted_dates = sorted(dates, key=lambda d: d.char_start)
        deduplicated = []
        skip_until = -1

        for date in sorted_dates:
            if date.char_start < skip_until:
                continue

            # Find best overlapping match
            best_date = date
            max_end = date.char_end

            for other in sorted_dates:
                if other.char_start >= max_end:
                    break
                if other.char_start >= date.char_start and other.char_start < date.char_end:
                    if other.confidence > best_date.confidence:
                        best_date = other
                    max_end = max(max_end, other.char_end)

            deduplicated.append(best_date)
            skip_until = max_end

        return deduplicated

    def _select_primary_date(self, dates: List[ExtractedDate]) -> Optional[ExtractedDate]:
        """Select the primary (most relevant) date."""
        if not dates:
            return None

        context_priority = {
            "deadline": 4, "event": 3, "publication": 2, "project": 1, "unknown": 0,
        }

        def score_date(date: ExtractedDate) -> float:
            score = context_priority.get(date.context, 0) * 10
            if date.day is not None:
                score += 20
            elif date.month is not None:
                score += 10
            score += date.confidence * 10
            return score

        return max(dates, key=score_date)


# Singleton instance
_extractor: Optional[DateExtractor] = None


def get_date_extractor() -> DateExtractor:
    """Get or create the singleton DateExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = DateExtractor()
    return _extractor


def extract_dates(
    text: str,
    reference_date: Optional[datetime] = None
) -> DateExtractionResult:
    """
    Convenience function to extract dates from text.

    Args:
        text: Text to extract dates from
        reference_date: Reference date for relative dates

    Returns:
        DateExtractionResult with all found dates
    """
    extractor = get_date_extractor()
    return extractor.extract(text, reference_date)


if __name__ == "__main__":
    test_texts = [
        "The proposal deadline is March 15, 2024.",
        "Our conference runs from March 20-25, 2024 in Chicago.",
        "This research was published in Q1 2024.",
        "The project started in Spring 2023 and ends December 2025.",
        "Submit by 03/15/24 for consideration.",
        "FY24 budget planning for the APTO project.",
    ]

    print("Testing Date Extractor\n" + "=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        result = extract_dates(text)

        if result.has_dates():
            for date in result.dates:
                print(f"  - {date.raw_text:20s} -> {date.normalized:12s} [{date.context}]")
            if result.primary_date:
                print(f"  Primary: {result.primary_date.raw_text}")
        else:
            print("  No dates found")
