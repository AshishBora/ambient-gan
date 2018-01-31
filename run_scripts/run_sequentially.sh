#!/bin/bash

# Make a temporary directory
tempdir=$(mktemp -d)


while true; do

    # Check if nothing is left. If yes, then break.
    all_scripts=$(ls ./scripts/*.sh)
    size=${#all_scripts}
    if [[ size -eq 0 ]]; then
        break
    fi

    # Move one file to tempdir
    for filename in ./scripts/*.sh; do
        mv $filename $tempdir
        break
    done

    # Run the one file we just moved
    for filename in $tempdir/*.sh; do
        $filename
        rm $filename
    done

done

# Clean the tempdir
rm -r $tempdir
