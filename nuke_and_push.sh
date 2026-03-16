#!/bin/bash
set -e
# ==============================================================================
# 🚀 FINAL REPOSITORY RESET SCRIPT — Purge & Deploy Clean Repo
# ==============================================================================
GH_USERNAME="code-with-idrees"
REPO_NAME="denoising-autoencoder-cifar10"
EMAIL="sleepingcatt01@gmail.com"

echo "----------------------------------------------------------------"
echo "🔐 STEP 1: Refreshing GitHub CLI Permissions"
echo "----------------------------------------------------------------"
gh auth refresh -h github.com -s delete_repo

echo "----------------------------------------------------------------"
echo "♻️  STEP 2: Deleting existing GitHub repository"
echo "----------------------------------------------------------------"
gh repo delete "$GH_USERNAME/$REPO_NAME" --confirm || true

echo "----------------------------------------------------------------"
echo "🧹 STEP 3: Rebuilding clean local Git history"
echo "----------------------------------------------------------------"
rm -rf .git
git init --initial-branch=main
git config user.name "$GH_USERNAME"
git config user.email "$EMAIL"

echo "----------------------------------------------------------------"
echo "🖼️  STEP 4: Fixing README image paths to GitHub raw CDN URLs"
echo "----------------------------------------------------------------"
python3 - <<'PYEOF'
import re

readme_path = "README.md"
base_url = "https://raw.githubusercontent.com/code-with-idrees/denoising-autoencoder-cifar10/main"

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix Markdown image links: ![alt](report/figures/...)
md_pattern = re.compile(r'!\[([^\]]*)\]\((report/figures/[^)]+)\)')
content, n1 = re.subn(md_pattern, lambda m: f'![{m.group(1)}]({base_url}/{m.group(2)})', content)

# Fix HTML img src: src="report/figures/..."
html_pattern = re.compile(r'src="(report/figures/[^"]+)"')
content, n2 = re.subn(html_pattern, lambda m: f'src="{base_url}/{m.group(1)}"', content)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"  ✅ Fixed {n1} markdown links and {n2} HTML src attributes.")
PYEOF

echo "----------------------------------------------------------------"
echo "🔨 STEP 5: Recreating repository on GitHub"
echo "----------------------------------------------------------------"
gh repo create "$REPO_NAME" --public --source=. --remote=origin

echo "----------------------------------------------------------------"
echo "📦 STEP 6: Pushing logical commits"
echo "----------------------------------------------------------------"
git add README.md LICENSE .gitignore requirements.txt
git commit -m "docs: finalized premium project organization and documentation"

git add src/ notebooks/
git commit -m "feat: core DAE implementation and statistical modules"

git add report/
git commit -m "report: integrated final academic LNCS report and 45+ figures"

git push -u origin main

echo "----------------------------------------------------------------"
echo "🏷️  STEP 7: Adding repository topics"
echo "----------------------------------------------------------------"
# --add-topic requires one topic at a time — comma-separated string does NOT work
gh repo edit "$GH_USERNAME/$REPO_NAME" \
  --add-topic "deep-learning" \
  --add-topic "computer-vision" \
  --add-topic "pytorch" \
  --add-topic "autoencoder" \
  --add-topic "denoising-autoencoder" \
  --add-topic "image-denoising" \
  --add-topic "cifar-10" \
  --add-topic "generative-ai"

echo "----------------------------------------------------------------"
echo "✨ SUCCESS! https://github.com/$GH_USERNAME/$REPO_NAME"
echo "----------------------------------------------------------------"
