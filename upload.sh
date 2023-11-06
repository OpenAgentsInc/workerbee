#!/bin/bash -e

if ! aws sts get-caller-identity --query "Account"; then
	aws sso login
fi


pushd dist
for f in gputopia-worker-*; do
    if [ -f "$f" ]; then
        aws s3 cp $f s3://gputopia/bin/ --acl public-read
    else
        aws s3 sync $f s3://gputopia/bin/$f --acl public-read
    fi    
done
popd
