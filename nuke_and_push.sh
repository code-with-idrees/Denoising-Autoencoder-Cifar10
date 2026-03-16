#!/bin/bash
set -e

# ==============================================================================
# 🚀 FINAL REPOSITORY RESET SCRIPT — Purge Stellar-Syntax & Deploy Well-Ordered Repo
# ==============================================================================

GH_USERNAME="code-with-idrees"
REPO_NAME="denoising-autoencoder-cifar10"
EMAIL="sleepingcatt01@gmail.com"

echo "----------------------------------------------------------------"
echo "🔐 STEP 1: Refreshing GitHub CLI Permissions"
echo "----------------------------------------------------------------"
gh auth refresh -h github.com -s delete_repo

echo "----------------------------------------------------------------"
echo "♻️ STEP 2: Deleting existing GitHub repository"
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
echo "🔨 STEP 4: Recreating repository on GitHub"
echo "----------------------------------------------------------------"
gh repo create "$REPO_NAME" --public --source=. --remote=origin

echo "----------------------------------------------------------------"
echo "📦 STEP 5: Pushing logical commits"
echo "----------------------------------------------------------------"
git add README.md LICENSE .gitignore requirements.txt
git commit -m "docs: finalized premium project organization and documentation"

git add src/ notebooks/
git commit -m "feat: core DAE implementation and statistical modules"

git add report/
git commit -m "report: integrated final academic LNCS report and 45+ figures"

git push -u origin main

echo "----------------------------------------------------------------"
echo "🏷️ STEP 6: Adding repository topics"
echo "----------------------------------------------------------------"
gh repo edit --add-topic "deep-learning,computer-vision,pytorch,autoencoder,denoising-autoencoder,image-denoising,cifar-10,generative-ai"

echo "----------------------------------------------------------------"
echo "✨ SUCCESS! Check your repo: https://github.com/$GH_USERNAME/$REPO_NAME"
echo "----------------------------------------------------------------"
