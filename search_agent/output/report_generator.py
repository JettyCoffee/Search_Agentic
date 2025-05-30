"""
Report generation utilities for creating human-readable outputs.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates human-readable reports from search results."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize report generator with configuration."""
        self.config = config or {}
        self.max_results_per_source = self.config.get("max_results_per_source", 5)
        self.include_metadata = self.config.get("include_metadata", True)
        self.include_sources = self.config.get("include_sources", True)
    
    def generate_text_report(self, formatted_output: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            formatted_output: Formatted search output
            
        Returns:
            Human-readable text report
        """
        try:
            report_parts = []
            
            # Header
            report_parts.append(self._generate_header(formatted_output))
            
            # Summary section
            report_parts.append(self._generate_summary_section(formatted_output))
            
            # Key findings section
            report_parts.append(self._generate_key_findings_section(formatted_output))
            
            # Detailed insights section
            report_parts.append(self._generate_insights_section(formatted_output))
            
            # Source results section
            if self.include_sources:
                report_parts.append(self._generate_sources_section(formatted_output))
            
            # Metadata section
            if self.include_metadata:
                report_parts.append(self._generate_metadata_section(formatted_output))
            
            # Footer
            report_parts.append(self._generate_footer(formatted_output))
            
            return "\n\n".join(report_parts)
            
        except Exception as e:
            logger.error(f"Text report generation failed: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def generate_markdown_report(self, formatted_output: Dict[str, Any]) -> str:
        """
        Generate a Markdown-formatted report.
        
        Args:
            formatted_output: Formatted search output
            
        Returns:
            Markdown-formatted report
        """
        try:
            report_parts = []
            
            # Title
            query = formatted_output.get("query", "Search Query")
            report_parts.append(f"# Search Results: {query}")
            
            # Summary
            synthesis = formatted_output.get("synthesis", {})
            summary = synthesis.get("summary", "No summary available")
            confidence = synthesis.get("confidence_level", "medium")
            
            report_parts.append("## Executive Summary")
            report_parts.append(f"**Confidence Level:** {confidence.title()}")
            report_parts.append(summary)
            
            # Key Findings
            key_findings = synthesis.get("key_findings", [])
            if key_findings:
                report_parts.append("## Key Findings")
                for i, finding in enumerate(key_findings, 1):
                    report_parts.append(f"{i}. {finding}")
            
            # Detailed Insights
            self._add_markdown_insights(report_parts, synthesis)
            
            # Search Results by Source
            if self.include_sources:
                self._add_markdown_sources(report_parts, formatted_output)
            
            # Search Metadata
            if self.include_metadata:
                self._add_markdown_metadata(report_parts, formatted_output)
            
            # Follow-up Suggestions
            followup = synthesis.get("suggested_followup", [])
            if followup:
                report_parts.append("## Suggested Follow-up Questions")
                for i, question in enumerate(followup, 1):
                    report_parts.append(f"{i}. {question}")
            
            return "\n\n".join(report_parts)
            
        except Exception as e:
            logger.error(f"Markdown report generation failed: {str(e)}")
            return f"# Error\n\nError generating report: {str(e)}"
    
    def save_report(self, formatted_output: Dict[str, Any], filepath: str, format_type: str = "markdown") -> bool:
        """
        Save report to file.
        
        Args:
            formatted_output: Formatted search output
            filepath: Path to save the report
            format_type: Report format ("markdown" or "text")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format_type.lower() == "markdown":
                content = self.generate_markdown_report(formatted_output)
            else:
                content = self.generate_text_report(formatted_output)
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {str(e)}")
            return False
    
    def generate_quick_summary(self, formatted_output: Dict[str, Any], max_length: int = 500) -> str:
        """
        Generate a quick summary for immediate consumption.
        
        Args:
            formatted_output: Formatted search output
            max_length: Maximum length of summary
            
        Returns:
            Quick summary text
        """
        try:
            synthesis = formatted_output.get("synthesis", {})
            summary = synthesis.get("summary", "No summary available")
            
            # Truncate if too long
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            # Add key stats
            metadata = formatted_output.get("metadata", {})
            total_results = metadata.get("total_results", 0)
            sources_count = len(metadata.get("sources_searched", []))
            
            stats = f"\n\n📊 Found {total_results} results from {sources_count} sources"
            
            return summary + stats
            
        except Exception as e:
            logger.error(f"Quick summary generation failed: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _generate_header(self, formatted_output: Dict[str, Any]) -> str:
        """Generate report header."""
        query = formatted_output.get("query", "Unknown Query")
        timestamp = formatted_output.get("metadata", {}).get("timestamp", datetime.now().timestamp())
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""SEARCH AGENT REPORT
{'=' * 50}
Query: {query}
Generated: {date_str}
{'=' * 50}"""
    
    def _generate_summary_section(self, formatted_output: Dict[str, Any]) -> str:
        """Generate summary section."""
        synthesis = formatted_output.get("synthesis", {})
        summary = synthesis.get("summary", "No summary available")
        confidence = synthesis.get("confidence_level", "medium")
        
        return f"""EXECUTIVE SUMMARY
{'-' * 20}
Confidence Level: {confidence.title()}

{summary}"""
    
    def _generate_key_findings_section(self, formatted_output: Dict[str, Any]) -> str:
        """Generate key findings section."""
        synthesis = formatted_output.get("synthesis", {})
        key_findings = synthesis.get("key_findings", [])
        
        if not key_findings:
            return "KEY FINDINGS\n" + "-" * 20 + "\nNo key findings identified."
        
        findings_text = "KEY FINDINGS\n" + "-" * 20
        for i, finding in enumerate(key_findings, 1):
            findings_text += f"\n{i}. {finding}"
        
        return findings_text
    
    def _generate_insights_section(self, formatted_output: Dict[str, Any]) -> str:
        """Generate detailed insights section."""
        synthesis = formatted_output.get("synthesis", {})
        sections = []
        
        # Academic insights
        academic = synthesis.get("academic_insights")
        if academic:
            sections.append(f"Academic Research:\n{academic}")
        
        # General information
        general = synthesis.get("general_information")
        if general:
            sections.append(f"General Information:\n{general}")
        
        # Background context
        background = synthesis.get("background_context")
        if background:
            sections.append(f"Background Context:\n{background}")
        
        # Limitations
        limitations = synthesis.get("limitations")
        if limitations:
            sections.append(f"Limitations:\n{limitations}")
        
        if not sections:
            return "DETAILED INSIGHTS\n" + "-" * 20 + "\nNo detailed insights available."
        
        return "DETAILED INSIGHTS\n" + "-" * 20 + "\n\n" + "\n\n".join(sections)
    
    def _generate_sources_section(self, formatted_output: Dict[str, Any]) -> str:
        """Generate sources section."""
        search_results = formatted_output.get("search_results", {})
        
        if not search_results:
            return "SOURCES\n" + "-" * 20 + "\nNo sources found."
        
        sources_text = "SOURCES\n" + "-" * 20
        
        for source, results in search_results.items():
            if not results:
                continue
            
            sources_text += f"\n\n{source.upper()} ({len(results)} results):"
            
            for i, result in enumerate(results[:self.max_results_per_source], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                
                sources_text += f"\n{i}. {title}"
                if url:
                    sources_text += f"\n   URL: {url}"
                if snippet:
                    snippet_short = snippet[:150] + "..." if len(snippet) > 150 else snippet
                    sources_text += f"\n   {snippet_short}"
        
        return sources_text
    
    def _generate_metadata_section(self, formatted_output: Dict[str, Any]) -> str:
        """Generate metadata section."""
        metadata = formatted_output.get("metadata", {})
        
        execution_time = metadata.get("execution_time_seconds", 0)
        total_results = metadata.get("total_results", 0)
        sources_searched = metadata.get("sources_searched", [])
        errors = metadata.get("errors", [])
        context_used = metadata.get("context_used", False)
        
        metadata_text = f"""SEARCH METADATA
{'-' * 20}
Execution Time: {execution_time:.2f} seconds
Total Results: {total_results}
Sources Searched: {', '.join(sources_searched)}
Context Used: {'Yes' if context_used else 'No'}"""
        
        if errors:
            metadata_text += f"\nErrors Encountered: {len(errors)}"
            for error in errors[:3]:  # Show first 3 errors
                metadata_text += f"\n  - {error}"
        
        return metadata_text
    
    def _generate_footer(self, formatted_output: Dict[str, Any]) -> str:
        """Generate report footer."""
        return f"""{'=' * 50}
End of Report
Generated by Search Agent Framework
{'=' * 50}"""
    
    def _add_markdown_insights(self, report_parts: List[str], synthesis: Dict[str, Any]) -> None:
        """Add insights sections to markdown report."""
        academic = synthesis.get("academic_insights")
        if academic:
            report_parts.append("### Academic Research")
            report_parts.append(academic)
        
        general = synthesis.get("general_information")
        if general:
            report_parts.append("### General Information")
            report_parts.append(general)
        
        background = synthesis.get("background_context")
        if background:
            report_parts.append("### Background Context")
            report_parts.append(background)
        
        limitations = synthesis.get("limitations")
        if limitations:
            report_parts.append("### Limitations")
            report_parts.append(limitations)
    
    def _add_markdown_sources(self, report_parts: List[str], formatted_output: Dict[str, Any]) -> None:
        """Add sources section to markdown report."""
        search_results = formatted_output.get("search_results", {})
        
        if not search_results:
            return
        
        report_parts.append("## Sources")
        
        for source, results in search_results.items():
            if not results:
                continue
            
            report_parts.append(f"### {source.title()} ({len(results)} results)")
            
            for i, result in enumerate(results[:self.max_results_per_source], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                
                if url:
                    report_parts.append(f"{i}. **[{title}]({url})**")
                else:
                    report_parts.append(f"{i}. **{title}**")
                
                if snippet:
                    snippet_short = snippet[:200] + "..." if len(snippet) > 200 else snippet
                    report_parts.append(f"   {snippet_short}")
    
    def _add_markdown_metadata(self, report_parts: List[str], formatted_output: Dict[str, Any]) -> None:
        """Add metadata section to markdown report."""
        metadata = formatted_output.get("metadata", {})
        
        execution_time = metadata.get("execution_time_seconds", 0)
        total_results = metadata.get("total_results", 0)
        sources_searched = metadata.get("sources_searched", [])
        errors = metadata.get("errors", [])
        context_used = metadata.get("context_used", False)
        
        report_parts.append("## Search Metadata")
        report_parts.append(f"- **Execution Time:** {execution_time:.2f} seconds")
        report_parts.append(f"- **Total Results:** {total_results}")
        report_parts.append(f"- **Sources Searched:** {', '.join(sources_searched)}")
        report_parts.append(f"- **Context Used:** {'Yes' if context_used else 'No'}")
        
        if errors:
            report_parts.append(f"- **Errors:** {len(errors)} encountered")
            for error in errors[:3]:
                report_parts.append(f"  - {error}")
