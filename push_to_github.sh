#!/bin/bash
# ================================================================
#  push_to_github.sh
#  Pushes all 4 AI/ML projects to salwabatool's GitHub account
#  Usage: bash push_to_github.sh
# ================================================================

set -e  # exit on any error

GITHUB_USERNAME="salwabatool"
REPO_NAME="ai-ml-projects"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   🚀  AI/ML Projects → GitHub Push Script       ║"
echo "║   User: $GITHUB_USERNAME                          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check git is installed ──────────────────────────────
if ! command -v git &>/dev/null; then
  echo "❌  git not found. Install it from https://git-scm.com/"
  exit 1
fi

# ── Step 2: Configure git identity (only if not already set) ────
if [ -z "$(git config --global user.email)" ]; then
  read -rp "  Enter your GitHub email: " GIT_EMAIL
  git config --global user.email "$GIT_EMAIL"
  git config --global user.name "$GITHUB_USERNAME"
fi

# ── Step 3: Initialise the repo ─────────────────────────────────
echo "  📁  Initialising git repository …"
git init
git checkout -b main 2>/dev/null || git checkout main

# ── Step 4: Add .gitignore ───────────────────────────────────────
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.env
venv/
.venv/
*.pth
*.pkl
*.pt
data/
*.png
*.jpg
*.jpeg
!README.md
EOF

# ── Step 5: Stage and commit ─────────────────────────────────────
echo "  📝  Staging all files …"
git add .
git commit -m "🤖 Add 4 AI/ML projects: NLP, CV, ML, Deep Learning

Projects included:
- 1-nlp-text-summarizer: BART + DistilBERT + extractive summarizer
- 2-computer-vision-classifier: ResNet-50 transfer learning + Grad-CAM
- 3-ml-house-price-predictor: 7 models compared (XGBoost wins)
- 4-deep-learning-cnn-mnist: ResNet with early stopping + t-SNE"

# ── Step 6: Add remote & push ────────────────────────────────────
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""
echo "  🌐  Remote URL: $REMOTE_URL"
echo ""
echo "  ⚠️   BEFORE RUNNING THIS SCRIPT:"
echo "       1. Create a NEW repo on GitHub named: $REPO_NAME"
echo "          → https://github.com/new"
echo "       2. Leave it EMPTY (no README, no .gitignore)"
echo "       3. Then press Enter here …"
read -rp "  Press Enter when repo is created: "

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo ""
echo "  📤  Pushing to GitHub … (you may be prompted for credentials)"
echo "      Tip: Use a Personal Access Token (PAT) as your password."
echo "      Generate one at: https://github.com/settings/tokens"
echo ""

git push -u origin main

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅  All 4 projects pushed successfully!         ║"
echo "║  🔗  https://github.com/$GITHUB_USERNAME/$REPO_NAME  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
