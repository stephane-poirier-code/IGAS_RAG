#!/usr/bin/env zsh

if [[ ${ZSH_EVAL_CONTEXT:-} != *:file* ]]; then
    print -u2 "Source this script in your terminal: source ./create_venv.sh"
    exit 1
fi

poetry env use python3.14 || return $?
poetry install || return $?

venv_path=$(poetry env info --path) || return $?
source "$venv_path/bin/activate" || return $?

print "Activated virtual environment: $VIRTUAL_ENV"
