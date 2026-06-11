# PyBaMM Solvers Test Suite

This directory contains the standardized test suite for the pybammsolvers package.

## Running Tests

### Using Nox (Recommended)

```bash
# Run tests through nox (handles dependencies)
nox -s test

# Run with specific Python version
nox -s test --python 3.13
```

### Using Pytest

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_module.py

# Run specific test class
pytest tests/test_vectors.py::TestVectorNdArrayBasic

# Run specific test
pytest tests/test_module.py::TestImport::test_pybammsolvers_import
```

### Using Test Markers

Tests are categorized with markers for selective execution:

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Combine markers
pytest -m "not slow and not integration"
```