# Contributing to Audio AI Fundamentals

First off, thank you for considering contributing to Audio AI Fundamentals! It's people like you that make this project a great tool for cultural heritage preservation.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to oggettosonoro@gmail.com.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

**Bug Report Template:**
```markdown
**Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.11]
 - Dependencies Version: [run `pip freeze`]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Provide specific examples** to demonstrate the steps
- **Describe the current behavior** and explain which behavior you expected to see instead
- **Explain why this enhancement would be useful** to most users
- **List some other projects** where this enhancement exists (if applicable)

### Adding New Cultural Institution Integrations

We especially welcome contributions that add support for new cultural institutions:

1. **Research the Institution**
   - Understand their audio archives
   - Identify specific needs and workflows
   - Document API/integration requirements

2. **Create Integration Module**
   ```python
   # week4-cultural-ai/integrations/institution_name.py
   class InstitutionNameIntegration:
       """Integration for [Institution Name]"""
       
       def __init__(self, config):
           self.config = config
           
       def process_archive(self, audio_files):
           """Process institution-specific archive format"""
           pass
   ```

3. **Add Documentation**
   - Create `docs/integrations/institution_name.md`
   - Include setup instructions
   - Provide usage examples
   - Document any special requirements

### Improving AI Models

Contributions to improve our cultural heritage AI models are highly valued:

1. **Dataset Contributions**
   - Add labeled cultural audio samples
   - Improve existing classifications
   - Expand regional/dialect coverage

2. **Model Improvements**
   - Optimize existing algorithms
   - Add new classification categories
   - Improve accuracy metrics

3. **Testing & Validation**
   - Add test cases for edge cases
   - Validate on new datasets
   - Performance benchmarking

## Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/audio-ai-fundamentals.git
   cd audio-ai-fundamentals
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Run Tests**
   ```bash
   pytest tests/
   ```

## Pull Request Process

1. **Ensure your code follows our style guidelines** (see below)
2. **Update the README.md** with details of changes if applicable
3. **Add tests** for new functionality
4. **Ensure all tests pass** by running `pytest`
5. **Update documentation** as needed
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork** and submit a pull request

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] New institution integration

## How Has This Been Tested?
Describe the tests that you ran to verify your changes.

## Checklist:
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
```

## Style Guidelines

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
class AudioProcessor:
    """
    Process audio files for cultural heritage applications.
    
    This class handles various audio processing tasks specific to
    cultural institution needs.
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz
        channels (int): Number of audio channels
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        
    def process_heritage_audio(self, audio_path: str) -> dict:
        """
        Process heritage audio file and extract metadata.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing audio metadata and analysis results
            
        Raises:
            ValueError: If audio file is invalid
        """
        # Implementation here
        pass
```

### Documentation Style

- Use Google-style docstrings
- Include type hints for all functions
- Provide examples in docstrings for complex functions
- Keep README files updated and comprehensive

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add RAI Teche integration module
fix: correct audio normalization in batch processor
docs: update installation instructions for Windows
test: add unit tests for heritage classifier
refactor: optimize FFT computation for large files
```

## Testing Guidelines

### Unit Tests
```python
# tests/test_audio_processor.py
import pytest
from week1_audio_visualizer import AudioProcessor

class TestAudioProcessor:
    def test_sample_rate_validation(self):
        """Test that invalid sample rates raise ValueError"""
        with pytest.raises(ValueError):
            AudioProcessor(sample_rate=-1)
            
    def test_process_heritage_audio(self, sample_audio_file):
        """Test heritage audio processing"""
        processor = AudioProcessor()
        result = processor.process_heritage_audio(sample_audio_file)
        assert 'metadata' in result
        assert result['metadata']['duration'] > 0
```

### Integration Tests
Test interactions between components, especially for cultural institution integrations.

### Performance Tests
Ensure code meets performance requirements for production use.

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Email**: oggettosonoro@gmail.com for direct questions
- **LinkedIn**: Connect at [linkedin.com/in/mainenti](https://linkedin.com/in/mainenti)

### Weekly Community Calls

We hold weekly community calls every Thursday at 16:00 CET to discuss:
- Ongoing development
- New institution integrations
- Technical challenges
- Future roadmap

### Recognition

Contributors who make significant contributions will be:
- Listed in our CONTRIBUTORS.md file
- Mentioned in release notes
- Invited to become project maintainers

## Thank You!

Your contributions to preserving cultural heritage through technology are invaluable. Together, we're building tools that will help preserve and make accessible the rich audio heritage of cultural institutions worldwide.

🎭 Happy coding and thank you for contributing to cultural preservation! 🏛️