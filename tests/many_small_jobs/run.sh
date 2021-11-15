#! /bin/bash
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

(
    cd $PARENT_PATH

    NUM_JOBS=$1
    PROCS=()
    for (( i = 0; i < $NUM_JOBS; ++i )); do
        python job_pack.py &
        PROCS+=($!)

        python job_spread.py &
        PROCS+=($!)
    done

    for proc in ${PROCS[@]}; do
        wait $proc
    done
)