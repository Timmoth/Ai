#!/bin/bash
set -e

# Save current directory (root)
ROOT_DIR=$(pwd)

# Find all publish.sh scripts under src/ (only one level deep)
PROJECT_SCRIPTS=$(find "$ROOT_DIR/src" -maxdepth 2 -type f -name "publish.sh")

if [ -z "$PROJECT_SCRIPTS" ]; then
    echo "No publish.sh scripts found under src/"
    exit 1
fi

for script in $PROJECT_SCRIPTS; do
    proj_dir=$(dirname "$script")
    echo "Publishing $proj_dir..."
    (
        cd "$proj_dir"
        if [ -x ./publish.sh ]; then
            ./publish.sh
        else
            echo "Warning: $proj_dir/publish.sh is not executable"
        fi
    )
done

echo "All projects published successfully!"
