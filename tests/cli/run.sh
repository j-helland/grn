#! /bin/bash
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
(
    cd $PARENT_PATH
    grn-run -d "python job.py"
)