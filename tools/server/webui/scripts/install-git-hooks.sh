#!/bin/bash

# Script to install pre-commit and pre-push hooks for webui
# Pre-commit: formats code and runs checks
# Pre-push: always builds webui and stages tools/server/public/

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"
PRE_PUSH_HOOK="$REPO_ROOT/.git/hooks/pre-push"

echo "Installing pre-commit and pre-push hooks for webui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Check if there are any changes in the webui directory
if git diff --cached --name-only | grep -q "^tools/server/webui/"; then
    echo "Formatting and checking webui code..."
    
    # Change to webui directory and run format
    cd tools/server/webui
    
    # Check if npm is available and package.json exists
    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in tools/server/webui"
        exit 1
    fi
    
    # Run the format command
    npm run format

    # Check if format command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run format failed"
        exit 1
    fi

    # Run the lint command
    npm run lint
    
    # Check if lint command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run lint failed"
        exit 1
    fi

    # Run the check command
    npm run check
    
    # Check if check command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run check failed"
        exit 1
    fi

    # Go back to repo root
    cd ../../..
    
    echo "✅ Webui code formatted and checked successfully"
fi

exit 0
EOF

# Create the pre-push hook
cat > "$PRE_PUSH_HOOK" << 'EOF'
#!/bin/bash

# Skip if already running from our own recursive push
if [ "$WEBUI_PRE_PUSH_RUNNING" = "1" ]; then
    exit 0
fi

REMOTE="$1"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

echo "Pre-push: building webui..."

# Change to webui directory
cd tools/server/webui

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found in tools/server/webui"
    exit 1
fi

# Always run the build
npm run build

if [ $? -ne 0 ]; then
    echo "❌ npm run build failed"
    exit 1
fi

# Go back to repo root
cd "$REPO_ROOT"

# Stage all build output in tools/server/public/
git add tools/server/public/

# Check if the build produced any changes compared to what's already committed
if ! git diff --cached --quiet -- tools/server/public/; then
    echo ""
    echo "⚠️  Build output in tools/server/public/ has changed."
    echo "   Committing updated build output..."
    git commit -m "server: update webui build output"

    # Push the updated branch ourselves (env var prevents recursion)
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "   Pushing $CURRENT_BRANCH to $REMOTE..."
    WEBUI_PRE_PUSH_RUNNING=1 git push "$REMOTE" "$CURRENT_BRANCH"
    PUSH_RESULT=$?

    if [ $PUSH_RESULT -eq 0 ]; then
        echo "✅ Build output committed and pushed successfully."
    else
        echo "❌ Push failed"
    fi

    # Cancel the original push — it has a stale SHA, we already pushed the updated one
    exit 1
fi

echo "✅ Build output is up-to-date, pushing..."
exit 0
EOF

# Make hooks executable
chmod +x "$PRE_COMMIT_HOOK"
chmod +x "$PRE_PUSH_HOOK"

if [ $? -eq 0 ]; then
    echo "✅ Git hooks installed successfully!"
    echo "   Pre-commit: $PRE_COMMIT_HOOK"
    echo "   Pre-push:   $PRE_PUSH_HOOK"
    echo ""
    echo "The hooks will automatically:"
    echo "  • Format and check webui code before commits (pre-commit)"
    echo "  • Build webui and stage tools/server/public/ before pushes (pre-push)"
    echo "  • If build output changed, commit it and abort push (just push again)"
else
    echo "❌ Failed to make hooks executable"
    exit 1
fi
