"""
Data validation and preprocessing utilities.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from urllib.parse import urlparse
from pydantic import BaseModel, ValidationError, validator
import html
import unicodedata

logger = logging.getLogger(__name__)


class QueryValidator:
    """Validates and preprocesses search queries."""
    
    @staticmethod
    def validate_query(query: str) -> Dict[str, Any]:
        """
        Validate and analyze a search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Dictionary with validation results and processed query
        """
        result = {
            "is_valid": False,
            "processed_query": "",
            "original_query": query,
            "issues": [],
            "suggestions": [],
            "metadata": {}
        }
        
        try:
            # Basic validation
            if not query or not isinstance(query, str):
                result["issues"].append("Query is empty or not a string")
                return result
            
            # Clean and preprocess
            processed_query = QueryValidator.preprocess_query(query)
            
            if not processed_query.strip():
                result["issues"].append("Query is empty after preprocessing")
                return result
            
            result["processed_query"] = processed_query
            result["is_valid"] = True
            
            # Analyze query characteristics
            metadata = QueryValidator._analyze_query(processed_query)
            result["metadata"] = metadata
            
            # Generate suggestions
            suggestions = QueryValidator._generate_suggestions(processed_query, metadata)
            result["suggestions"] = suggestions
            
            # Check for potential issues
            issues = QueryValidator._check_issues(processed_query, metadata)
            result["issues"] = issues
            
            return result
            
        except Exception as e:
            logger.error(f"Query validation failed: {str(e)}")
            result["issues"].append(f"Validation error: {str(e)}")
            return result
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """
        Preprocess a search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Preprocessed query
        """
        if not query:
            return ""
        
        # Convert to string if not already
        query = str(query)
        
        # Normalize Unicode characters
        query = unicodedata.normalize('NFKC', query)
        
        # Decode HTML entities
        query = html.unescape(query)
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Strip leading/trailing whitespace
        query = query.strip()
        
        # Remove control characters
        query = ''.join(char for char in query if unicodedata.category(char)[0] != 'C')
        
        # Limit length (reasonable maximum)
        if len(query) > 500:
            query = query[:497] + "..."
            logger.warning("Query truncated due to excessive length")
        
        return query
    
    @staticmethod
    def _analyze_query(query: str) -> Dict[str, Any]:
        """Analyze query characteristics."""
        metadata = {
            "length": len(query),
            "word_count": len(query.split()),
            "has_quotes": '"' in query,
            "has_operators": any(op in query.lower() for op in ['and', 'or', 'not', '+', '-']),
            "has_wildcards": any(char in query for char in ['*', '?']),
            "is_question": query.strip().endswith('?'),
            "has_numbers": bool(re.search(r'\d', query)),
            "has_special_chars": bool(re.search(r'[!@#$%^&*()_+={}\[\]|\\:";\'<>,.?/~`]', query)),
            "language_hints": QueryValidator._detect_language_hints(query),
            "academic_indicators": QueryValidator._detect_academic_indicators(query),
            "temporal_indicators": QueryValidator._detect_temporal_indicators(query)
        }
        
        return metadata
    
    @staticmethod
    def _detect_language_hints(query: str) -> List[str]:
        """Detect language or domain hints in the query."""
        hints = []
        
        # Programming languages
        prog_languages = ['python', 'javascript', 'java', 'c++', 'r', 'sql', 'html', 'css']
        for lang in prog_languages:
            if lang.lower() in query.lower():
                hints.append(f"programming:{lang}")
        
        # Academic fields
        academic_fields = ['medicine', 'biology', 'physics', 'chemistry', 'computer science', 'mathematics']
        for field in academic_fields:
            if field.lower() in query.lower():
                hints.append(f"academic:{field}")
        
        # Scientific terms
        if re.search(r'\b(study|research|analysis|experiment|hypothesis)\b', query.lower()):
            hints.append("scientific")
        
        return hints
    
    @staticmethod
    def _detect_academic_indicators(query: str) -> List[str]:
        """Detect indicators that suggest academic/research focus."""
        indicators = []
        
        academic_terms = [
            'paper', 'study', 'research', 'analysis', 'review', 'meta-analysis',
            'systematic review', 'clinical trial', 'experiment', 'hypothesis',
            'methodology', 'findings', 'results', 'conclusion', 'abstract',
            'doi', 'pubmed', 'arxiv', 'journal', 'publication'
        ]
        
        for term in academic_terms:
            if term.lower() in query.lower():
                indicators.append(term)
        
        return indicators
    
    @staticmethod
    def _detect_temporal_indicators(query: str) -> List[str]:
        """Detect temporal indicators in the query."""
        indicators = []
        
        # Year patterns
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if years:
            indicators.extend([f"year:{year}" for year in years])
        
        # Temporal terms
        temporal_terms = [
            'recent', 'latest', 'current', 'new', 'old', 'historical',
            'past', 'future', 'today', 'yesterday', 'tomorrow',
            'last year', 'this year', 'next year'
        ]
        
        for term in temporal_terms:
            if term.lower() in query.lower():
                indicators.append(term)
        
        return indicators
    
    @staticmethod
    def _generate_suggestions(query: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate query improvement suggestions."""
        suggestions = []
        
        if metadata["word_count"] < 3:
            suggestions.append("Consider adding more specific terms to improve search results")
        
        if metadata["word_count"] > 15:
            suggestions.append("Consider breaking down into more focused searches")
        
        if not metadata["has_quotes"] and metadata["word_count"] > 5:
            suggestions.append("Consider using quotes for exact phrases")
        
        if metadata["academic_indicators"] and not any("academic" in hint for hint in metadata["language_hints"]):
            suggestions.append("Consider searching academic databases specifically")
        
        if metadata["is_question"]:
            suggestions.append("Try converting question to keyword-based search")
        
        return suggestions
    
    @staticmethod
    def _check_issues(query: str, metadata: Dict[str, Any]) -> List[str]:
        """Check for potential query issues."""
        issues = []
        
        if metadata["length"] < 3:
            issues.append("Query is very short")
        
        if metadata["word_count"] == 1 and len(query) < 3:
            issues.append("Single character queries may not be effective")
        
        # Check for common stop words only
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        query_words = query.lower().split()
        if all(word in stop_words for word in query_words):
            issues.append("Query consists only of common stop words")
        
        return issues


class ResultValidator:
    """Validates and cleans search results."""
    
    @staticmethod
    def validate_search_result(result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Validate and clean a search result.
        
        Args:
            result: Raw search result
            source: Source name (e.g., 'google', 'arxiv')
            
        Returns:
            Validated and cleaned result
        """
        validated_result = {
            "title": "",
            "url": "",
            "snippet": "",
            "source": source,
            "metadata": {},
            "validation_issues": []
        }
        
        try:
            # Validate title
            title = ResultValidator._clean_text(result.get("title", ""))
            if title:
                validated_result["title"] = title
            else:
                validated_result["validation_issues"].append("Missing or invalid title")
                validated_result["title"] = "Untitled"
            
            # Validate URL
            url = result.get("url", "")
            if ResultValidator._is_valid_url(url):
                validated_result["url"] = url
            else:
                validated_result["validation_issues"].append("Invalid or missing URL")
            
            # Validate snippet/description
            snippet = ResultValidator._clean_text(
                result.get("snippet", result.get("abstract", result.get("description", "")))
            )
            validated_result["snippet"] = snippet
            
            # Extract and validate metadata
            metadata = ResultValidator._extract_metadata(result, source)
            validated_result["metadata"] = metadata
            
            # Source-specific validation
            source_issues = ResultValidator._validate_source_specific(result, source)
            validated_result["validation_issues"].extend(source_issues)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Result validation failed for {source}: {str(e)}")
            validated_result["validation_issues"].append(f"Validation error: {str(e)}")
            return validated_result
    
    @staticmethod
    def validate_search_results(results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        Validate a list of search results.
        
        Args:
            results: List of raw search results
            source: Source name
            
        Returns:
            List of validated results
        """
        if not isinstance(results, list):
            logger.warning(f"Results for {source} is not a list: {type(results)}")
            return []
        
        validated_results = []
        for i, result in enumerate(results):
            try:
                validated_result = ResultValidator.validate_search_result(result, source)
                validated_results.append(validated_result)
            except Exception as e:
                logger.error(f"Failed to validate result {i} for {source}: {str(e)}")
                continue
        
        return validated_results
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip
        text = text.strip()
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char in '\n\t' or unicodedata.category(char)[0] != 'C')
        
        return text
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    @staticmethod
    def _extract_metadata(result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Extract and clean metadata from result."""
        metadata = {}
        
        # Common metadata fields
        common_fields = [
            'published_date', 'authors', 'doi', 'venue', 'citations',
            'categories', 'keywords', 'language', 'type'
        ]
        
        for field in common_fields:
            if field in result:
                value = result[field]
                if value is not None:
                    metadata[field] = value
        
        # Source-specific metadata
        if source == "arxiv":
            if "arxiv_id" in result:
                metadata["arxiv_id"] = result["arxiv_id"]
            if "primary_category" in result:
                metadata["primary_category"] = result["primary_category"]
        
        elif source == "semantic_scholar":
            if "paper_id" in result:
                metadata["paper_id"] = result["paper_id"]
            if "influential_citation_count" in result:
                metadata["influential_citation_count"] = result["influential_citation_count"]
        
        elif source in ["google", "brave"]:
            if "displayed_url" in result:
                metadata["displayed_url"] = result["displayed_url"]
            if "position" in result:
                metadata["position"] = result["position"]
        
        # Validate dates
        if "published_date" in metadata:
            metadata["published_date"] = ResultValidator._validate_date(metadata["published_date"])
        
        return metadata
    
    @staticmethod
    def _validate_date(date_value: Any) -> Optional[str]:
        """Validate and normalize date values."""
        if not date_value:
            return None
        
        try:
            # If already a datetime object
            if isinstance(date_value, datetime):
                return date_value.isoformat()
            
            # If string, try to parse
            if isinstance(date_value, str):
                # Try common formats
                formats = [
                    "%Y-%m-%d",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%Y"
                ]
                
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_value, fmt)
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
                
                # If no format matched, return as-is if it looks like a year
                if re.match(r'^\d{4}$', date_value):
                    return date_value
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def _validate_source_specific(result: Dict[str, Any], source: str) -> List[str]:
        """Perform source-specific validation."""
        issues = []
        
        if source == "arxiv":
            if "arxiv_id" not in result:
                issues.append("Missing ArXiv ID")
            
            # Validate ArXiv ID format
            arxiv_id = result.get("arxiv_id", "")
            if arxiv_id and not re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
                issues.append("Invalid ArXiv ID format")
        
        elif source == "semantic_scholar":
            if "paper_id" not in result:
                issues.append("Missing Semantic Scholar paper ID")
        
        elif source in ["google", "brave"]:
            # Check for ad indicators
            title = result.get("title", "").lower()
            if any(indicator in title for indicator in ["ad", "sponsored", "advertisement"]):
                issues.append("Possible advertisement")
        
        return issues


class DataCleaner:
    """General data cleaning utilities."""
    
    @staticmethod
    def remove_duplicates(results: List[Dict[str, Any]], key_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on key fields.
        
        Args:
            results: List of results
            key_fields: Fields to use for deduplication (default: ['title', 'url'])
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        if key_fields is None:
            key_fields = ['title', 'url']
        
        seen = set()
        unique_results = []
        
        for result in results:
            # Create key from specified fields
            key_parts = []
            for field in key_fields:
                value = result.get(field, "")
                if value:
                    # Normalize for comparison
                    normalized = value.lower().strip()
                    key_parts.append(normalized)
            
            key = tuple(key_parts)
            
            if key and key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        removed_count = len(results) - len(unique_results)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate results")
        
        return unique_results
    
    @staticmethod
    def filter_by_relevance(results: List[Dict[str, Any]], query: str, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter results by relevance to query.
        
        Args:
            results: List of results
            query: Original search query
            min_score: Minimum relevance score (0-1)
            
        Returns:
            Filtered results
        """
        if not results or not query:
            return results
        
        query_terms = set(query.lower().split())
        filtered_results = []
        
        for result in results:
            score = DataCleaner._calculate_relevance_score(result, query_terms)
            if score >= min_score:
                result["relevance_score"] = score
                filtered_results.append(result)
        
        # Sort by relevance score
        filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        removed_count = len(results) - len(filtered_results)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} low-relevance results")
        
        return filtered_results
    
    @staticmethod
    def _calculate_relevance_score(result: Dict[str, Any], query_terms: set) -> float:
        """Calculate relevance score for a result."""
        if not query_terms:
            return 1.0
        
        # Combine title and snippet for scoring
        text_fields = [
            result.get("title", ""),
            result.get("snippet", ""),
            result.get("abstract", "")
        ]
        
        combined_text = " ".join(text_fields).lower()
        text_words = set(combined_text.split())
        
        if not text_words:
            return 0.0
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(text_words))
        max_possible = len(query_terms)
        
        if max_possible == 0:
            return 1.0
        
        return overlap / max_possible
