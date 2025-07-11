# Git Workflow & GitHub Integration

## 🎯 Repository Setup

**GitHub Repository**: https://github.com/terrylica/nautilus-trader-workspace  
**Type**: Public repository  
**Description**: Complete NautilusTrader development workspace with strategies, backtesting, learning notes, and reference implementation
**Scope**: Entire workspace (not just nautilus_test subdirectory)

### Authentication Status ✅
- **GitHub CLI**: Authenticated as `terrylica`
- **Git Config**: 
  - User: `terrylica`
  - Email: `terry@eonlabs.com`
- **Token Scopes**: `gist`, `read:org`, `repo`

## 🔄 Development Workflow

### Daily Development Flow
```bash
# 1. Check status and pull latest changes
git status
git pull origin master

# 2. Create feature branch (optional for learning)
git checkout -b feature/new-strategy

# 3. Development cycle
make format && make lint && make typecheck && make test

# 4. Stage and commit changes
git add .
git commit -m "Add new EMA crossover strategy

- Implemented basic EMA cross strategy
- Added backtesting configuration
- Updated documentation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 5. Push to GitHub
git push origin master  # or feature branch
```

### Commit Message Format
```
<type>: <short description>

<detailed description>
- Bullet point changes
- More details as needed

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Commit Types**:
- `feat`: New feature (strategy, indicator, etc.)
- `fix`: Bug fix
- `docs`: Documentation updates
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

## 📋 GitHub CLI Commands

### Repository Management
```bash
# View repository details
gh repo view

# Open repository in browser
gh repo view --web

# Create new repository
gh repo create <name> --public --description "Description"

# Clone repository
gh repo clone terrylica/nautilus-trader-learning
```

### Issues & Projects
```bash
# Create issue
gh issue create --title "Implement RSI strategy" --body "Description"

# List issues
gh issue list

# View issue details
gh issue view <number>
```

### Pull Requests (for collaboration)
```bash
# Create pull request
gh pr create --title "Add new strategy" --body "Description"

# List pull requests
gh pr list

# Merge pull request
gh pr merge <number>
```

## 🌿 Branch Strategy

### Simple Learning Workflow
- **`master`**: Main development branch (your work)
- **Feature branches**: Optional for experimenting with strategies

### Example Branch Names
```bash
git checkout -b strategy/ema-cross
git checkout -b fix/backtest-config
git checkout -b docs/trading-guide
```

## 📁 Repository Structure

### What's Tracked
```
✅ Source code (src/)
✅ Tests (tests/)
✅ Documentation (learning_notes/)
✅ Examples (examples/)
✅ Configuration (pyproject.toml, Makefile)
✅ Dependencies (uv.lock)
```

### What's Ignored (.gitignore)
```
❌ Python cache (__pycache__)
❌ Virtual environments
❌ IDE files (.vscode/settings.json)
❌ Build artifacts
❌ Data files (large datasets)
❌ Credentials and secrets
```

## 🔧 Git Configuration

### Useful Git Aliases
```bash
# Add these to your git config
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --oneline --graph --decorate"
```

### Git Hooks (Future Enhancement)
- Pre-commit: Run `make lint` before commits
- Pre-push: Run `make test` before pushing

## 🚀 Integration Benefits

### Version Control Advantages
- ✅ **Backup**: All work automatically backed up to GitHub
- ✅ **History**: Complete change history for strategies
- ✅ **Collaboration**: Easy sharing with other developers
- ✅ **Rollback**: Ability to revert problematic changes
- ✅ **Branching**: Experiment safely with new ideas

### GitHub Features Available
- ✅ **Issues**: Track bugs and feature requests
- ✅ **Wiki**: Additional documentation
- ✅ **Releases**: Tag stable versions of strategies
- ✅ **Actions**: Automated testing (future enhancement)
- ✅ **Pages**: Host documentation websites

## 📊 Next Steps with Git

### Immediate Actions
1. Commit changes regularly (daily or after each feature)
2. Use descriptive commit messages
3. Keep repository organized and clean

### Future Enhancements
1. **GitHub Actions**: Automated testing on push
2. **Branch Protection**: Require tests to pass
3. **Code Review**: Use pull requests for major changes
4. **Releases**: Tag stable strategy versions

### Learning Resources
- Git documentation: https://git-scm.com/docs
- GitHub CLI: https://cli.github.com/manual/
- GitHub Guides: https://guides.github.com/

---

**Created**: 2025-07-11  
**Repository**: https://github.com/terrylica/nautilus-trader-learning  
**Status**: Fully configured and operational