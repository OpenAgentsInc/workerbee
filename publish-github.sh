#!/bin/bash -e
version=$1
if [ -z "$version" ]; then
    echo "usage: $0 <version>"
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "need a GITHUB_TOKEN env var or .env"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "uncommitted changes"
    exit 1
fi

if ! grep $version pyproject.toml; then
    echo "version $version not in pyproject.toml"
    exit 1
fi

git fetch --tags

if git tag | grep $version; then
    echo "already have a $version"
    exit 1
fi

rm -rf bin
aws s3 sync "s3://gputopia/bin/" bin/
githubrelease release arcadelabsinc/workerbee create "$version" --publish --name "gputopia-worker-$version" "bin/*[!z]"
