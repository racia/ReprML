#!/bin/bash

if (! -z $1); then
    DETERM="-determ"
fi

if (! -z ${DETERM}); then
    echo $DETERM
fi