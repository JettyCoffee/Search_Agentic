#!/usr/bin/env python3
"""
Test runner script for the search agent project.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the search agent project")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--format", action="store_true", help="Format code with black")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies first")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    success = True
    
    # Install dependencies if requested
    if args.install_deps:
        if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          "Installing dependencies"):
            success = False
    
    # Format code if requested
    if args.format:
        if not run_command([sys.executable, "-m", "black", "search_agent/", "tests/"], 
                          "Formatting code with black"):
            success = False
    
    # Run linting if requested
    if args.lint:
        if not run_command([sys.executable, "-m", "flake8", "search_agent/"], 
                          "Running flake8 linting"):
            success = False
        
        if not run_command([sys.executable, "-m", "mypy", "search_agent/"], 
                          "Running mypy type checking"):
            success = False
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend(["--cov=search_agent", "--cov-report=html", "--cov-report=term"])
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    # Select test types
    if args.unit and not args.integration:
        pytest_cmd.append("tests/unit/")
    elif args.integration and not args.unit:
        pytest_cmd.append("tests/integration/")
    elif not args.unit and not args.integration:
        pytest_cmd.append("tests/")
    else:
        pytest_cmd.append("tests/")
    
    # Run tests
    if not run_command(pytest_cmd, "Running tests"):
        success = False
    
    # Generate coverage report if requested
    if args.coverage:
        print(f"\nCoverage report generated in htmlcov/index.html")
    
    # Exit with appropriate code
    if success:
        print(f"\n{'='*60}")
        print("✅ All tests and checks passed!")
        print('='*60)
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("❌ Some tests or checks failed!")
        print('='*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
