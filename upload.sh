#!/bin/bash

aws s3 cp dist/gputopia-worker-* s3://gputopia/bin/ --acl public-read
