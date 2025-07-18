# Contributing to MCP-fMRI

Thank you for your interest in contributing to MCP-fMRI! This project aims to promote ethical neuroimaging research that emphasizes gender similarities and integrates cultural context. All contributions should align with these core principles.

## Code of Conduct

### Our Commitment
We are committed to creating an inclusive, respectful, and ethical research environment. All contributors must:

- Prioritize gender similarities over differences in analysis and interpretation
- Respect cultural context and avoid stereotyping
- Focus on individual variation rather than group generalizations
- Support evidence-based approaches to understanding cognition
- Never contribute to discrimination or stereotype reinforcement

### Expected Behavior
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Focus on constructive feedback
- Show empathy towards community members
- Prioritize scientific integrity and ethical considerations

## Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the issue tracker to report bugs
- Include detailed steps to reproduce
- Specify your environment (OS, Python version, etc.)
- Check if the bug has already been reported

### 💡 Feature Requests
- Propose new features that align with ethical guidelines
- Explain the use case and potential benefits
- Consider cultural sensitivity in your proposal
- Focus on similarity-enhancing features

### 🔧 Code Contributions
- Bug fixes
- New analysis methods (with ethical focus)
- Cultural context enhancements
- Bias detection improvements
- Documentation improvements
- Test coverage enhancements

### 📚 Documentation
- API documentation
- Tutorials and examples
- Ethical guidelines clarification
- Cultural context explanations
- Translation of documentation

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/mcp-fmri-package.git
cd mcp-fmri-package
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies
```bash
pip install -e .[dev]
```

### 4. Run Tests
```bash
pytest tests/
```

## Development Guidelines

### Ethical Code Requirements
All code contributions must:

1. **Emphasize Similarities**: Focus on detecting and highlighting similarities rather than differences
2. **Include Bias Detection**: Implement or enhance bias detection mechanisms
3. **Support Cultural Context**: Allow for cultural context integration
4. **Promote Individual Focus**: Emphasize individual variation over group patterns
5. **Maintain Transparency**: Include clear documentation and ethical considerations

### Code Style
- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/`
- Use type hints where appropriate
- Write descriptive variable and function names
- Include docstrings for all public functions

### Testing Requirements
- Write tests for all new functionality
- Maintain test coverage above 80%
- Include ethical compliance tests
- Test bias detection functionality
- Verify cultural context integration

### Example Ethical Test
```python
def test_similarity_emphasis():
    """Test that analysis emphasizes similarities."""
    analyzer = GenderSimilarityAnalyzer(ethical_guidelines=True)
    result = analyzer.analyze_similarities(test_data)
    
    # Ensure similarity metrics are prominently featured
    assert 'overall_similarity_index' in result
    assert result['similarity_focus'] is True
    assert result['individual_to_group_ratio'] > 1
```

## Submission Process

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow coding guidelines
- Write tests
- Update documentation
- Ensure ethical compliance

### 3. Test Your Changes
```bash
# Run all tests
pytest tests/

# Check code style
black --check src/
flake8 src/

# Test ethical compliance
python -c "from mcp_fmri.utils import check_ethical_compliance; print(check_ethical_compliance(your_config))"
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add similarity-focused analysis method"
```

**Commit Message Format:**
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `ethical:` ethical guideline improvements

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Pull Request Guidelines

### Required Information
Your pull request should include:

1. **Clear Description**: Explain what the PR does and why
2. **Ethical Considerations**: How does this align with ethical guidelines?
3. **Cultural Sensitivity**: Any cultural context considerations?
4. **Testing**: What tests were added/modified?
5. **Documentation**: What documentation was updated?
6. **Breaking Changes**: Any breaking changes?

### PR Template
```markdown
## Description
Brief description of changes

## Ethical Considerations
- [ ] Emphasizes similarities over differences
- [ ] Includes bias detection
- [ ] Supports cultural context
- [ ] Focuses on individual variation

## Testing
- [ ] New tests added
- [ ] All tests pass
- [ ] Ethical compliance verified

## Documentation
- [ ] Code documented
- [ ] Examples updated
- [ ] Ethical guidelines reflected

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Cultural sensitivity considered
- [ ] No discriminatory potential
```

## Review Process

### What Reviewers Look For
1. **Code Quality**: Clean, readable, well-documented code
2. **Ethical Compliance**: Alignment with ethical guidelines
3. **Cultural Sensitivity**: Appropriate cultural considerations
4. **Scientific Validity**: Sound methodology and interpretation
5. **Test Coverage**: Adequate testing of new functionality

### Review Criteria
- ✅ **Approved**: Meets all requirements
- 🔄 **Changes Requested**: Needs modifications
- ❌ **Rejected**: Doesn't align with project goals

## Specific Contribution Areas

### Bias Detection Enhancements
- Improve bias detection algorithms
- Add new bias metrics
- Enhance sensitivity to subtle biases
- Cross-cultural bias validation

### Cultural Context Expansion
- Add support for new cultural contexts
- Improve existing cultural models
- Validate cultural assumptions
- Collaborate with cultural experts

### Analysis Methods
- Develop similarity-focused algorithms
- Improve individual variation analysis
- Enhance interpretability
- Add validation methods

### Visualization Improvements
- Create similarity-focused visualizations
- Improve accessibility
- Add interactive features
- Enhance cultural representation

## Community Guidelines

### Communication
- Use GitHub issues for bug reports and feature requests
- Join discussions on GitHub Discussions
- Be respectful and constructive
- Focus on scientific and ethical merit

### Collaboration
- Acknowledge all contributors
- Share credit appropriately
- Collaborate across disciplines
- Respect diverse perspectives

## Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Acknowledged in release notes
- Credited in academic publications (where appropriate)
- Invited to participate in project governance

## Questions?

If you have questions about contributing:

1. Check the [FAQ](docs/faq.md)
2. Search existing [issues](https://github.com/Tatsuru-Kikuchi/mcp-fmri-package/issues)
3. Start a [discussion](https://github.com/Tatsuru-Kikuchi/mcp-fmri-package/discussions)
4. Contact the maintainers

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

**Remember**: Every contribution should advance our mission of ethical, culturally-sensitive neuroimaging research that emphasizes human similarities and individual potential over group generalizations.

Thank you for helping make MCP-fMRI a force for positive, ethical science! 🧠✨