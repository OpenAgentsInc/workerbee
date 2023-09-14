#!/bin/bash

for f in dist/gputopia-worker-*; do
    aws s3 cp $f s3://gputopia/bin/ --acl public-read
done
