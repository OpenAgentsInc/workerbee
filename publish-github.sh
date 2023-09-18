#!/bin/bash
version=$1
if [ -z "$version" ]; then
    echo usage: $0 <version>
fi
rm -rf bin
aws s3 cp s3://gputopia/bin/* bin/
githubrelease release arcadelabsinc/workerbee create 2.0.0 --publish --name "gputopia-worker-$version" "bin/*"
