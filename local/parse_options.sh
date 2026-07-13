#!/usr/bin/env bash
# Kaldi-style option parser.
# Usage: . ./local/parse_options.sh

for ((argpos=1; argpos<$#; argpos++)); do
  if [ "${!argpos}" == "--config" ]; then
    argpos_plus1=$((argpos+1))
    config=${!argpos_plus1}
    [ ! -r "$config" ] && echo "$0: missing config '$config'" && exit 1
    . "$config"
  fi
done

while true; do
  [ -z "${1:-}" ] && break
  case "$1" in
    --help|-h)
      if [ -z "${help_message:-}" ]; then
        echo "No help found." >&2
      else
        printf "%s\n" "$help_message" >&2
      fi
      exit 0
      ;;
    --*=*)
      echo "$0: options must be '--name value', got '$1'" >&2
      exit 1
      ;;
    --*)
      name="$(echo "$1" | sed 's/^--//' | sed 's/-/_/g')"
      eval '[ -z "${'"$name"'+xxx}" ]' && {
        echo "$0: invalid option $1" >&2
        exit 1
      }

      [ $# -lt 2 ] && {
        echo "$0: option $1 requires an argument" >&2
        exit 1
      }

      oldval="$(eval echo \$$name)"
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then
        was_bool=true
      else
        was_bool=false
      fi

      eval "$name=\"\$2\""

      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected true/false for $1, got '$2'" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

true
