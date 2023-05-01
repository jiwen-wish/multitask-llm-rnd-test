#!/bin/bash

BASENAME="${0##*/}"

usage () {
  if [ "${#@}" -ne 0 ]; then
    echo "* ${*}"
    echo
  fi
  cat <<ENDUSAGE
Usage:
${BASENAME} branch-name s3-bucket-name
ENDUSAGE

  exit 2
}


# Standard function to print an error and exit with a failing return code
error_exit () {
  echo "${BASENAME} - ${1}" >&2
  exit 1
}


[ $# -ne 2 ] && usage "Missing arguments"
if ! git show-ref --quiet refs/heads/$1 2>/dev/null 1>/dev/null
then
   echo "Branch don't exist!" 
   error_exit $1 
fi

echo "Archiving Brangh:$1"
bname=$(basename `git rev-parse --show-toplevel`)
if ! git archive --format=zip -o ${bname}.zip $1
then
  echo "Unable to archive branch:$1"
  exit 1
fi

echo "Uploading script to S3 bucket"
aws s3 cp ${bname}.zip $2
