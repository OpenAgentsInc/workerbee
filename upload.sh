#!/bin/bash -e

if ! aws sts get-caller-identity --query "Account"; then
	aws sso login
fi


pushd dist
for f in gputopia-worker-*; do
    aws s3 cp $f s3://gputopia/bin/ --acl public-read
done
popd
