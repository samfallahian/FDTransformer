#!/bin/zsh

# Script to recursively find files like 'name.0.pkl'
# and rename them to 'name.pkl'.
# It logs actions and provides per-directory feedback.
# Ran with:
# /Users/kkreth/PycharmProjects/cgan/Ordered_080_FixFileNames.zsh /Users/kkreth/PycharmProjects/data/all_data_ready_for_training /tmp/filefix.log

# Function to log messages with a timestamp
log_message() {
  # Ensure LOG_FILE is available in this function's scope
  echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

main() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <target_directory> [log_file_path]"
    echo "Example: $0 /path/to/your/files rename_decimal_zero.log"
    return 1
  fi

  local TARGET_DIR
  TARGET_DIR="$(realpath "$1")" # Resolve to absolute path
  LOG_FILE="${2:-rename_decimal_zero.log}" # Default log file name

  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory '$TARGET_DIR' not found or is not a directory."
    return 1
  fi

  echo "Attempting to log to: $(realpath "$LOG_FILE" 2>/dev/null || echo "$LOG_FILE")"
  touch "$LOG_FILE" &>/dev/null
  if [[ $? -ne 0 ]]; then
    local log_dir
    log_dir="$(dirname "$LOG_FILE")"
    if [[ ! -w "$log_dir" ]]; then
        echo "Error: Cannot write to log file '$LOG_FILE'. Directory '$log_dir' may not be writable or log file is not accessible."
        return 1
    fi
    echo "Warning: Could not touch log file '$LOG_FILE', but will attempt to write."
  fi

  log_message "Script started. Target directory: '$TARGET_DIR'. Task: Rename '*.0.pkl' to '*.pkl'."
  echo "Processing files in '$TARGET_DIR' and its subdirectories."
  echo "Looking for files ending in '.0.pkl' to rename them by removing '.0'."
  echo "Logging operations to '$LOG_FILE'."
  echo # Blank line for readability

  find "$TARGET_DIR" -type d -print0 | sort -z | while IFS= read -r -d $'\0' current_dir; do
    local renamed_in_dir_count=0
    echo "Scanning directory: $current_dir"

    # Find files matching the pattern '*.0.pkl' directly within current_dir
    find "$current_dir" -maxdepth 1 -type f -name '*.0.pkl' -print0 | while IFS= read -r -d $'\0' original_file_path; do
      local original_filename
      original_filename="${original_file_path##*/}" # Extract filename

      # Construct the new filename by removing '.0' before '.pkl'
      # e.g., "1.0.pkl" becomes "1.pkl"
      # e.g., "data.test.0.pkl" becomes "data.test.pkl"
      local base_part="${original_filename%.0.pkl}" # Removes the trailing '.0.pkl'
      local new_basename="${base_part}.pkl"         # Adds back '.pkl'

      if [[ -z "$base_part" ]] || [[ "$new_basename" == ".pkl" ]] ; then
        log_message "SKIPPING: Could not derive valid new name for '$original_file_path'. Original: '$original_filename', derived base: '$base_part'."
        continue
      fi

      local dir_of_file
      dir_of_file="${original_file_path%/*}"
      local new_file_path="${dir_of_file}/${new_basename}"

      if [[ "$original_file_path" == "$new_file_path" ]]; then
        # This case should ideally not be hit if find pattern and rename logic are correct
        log_message "INFO: File '$original_file_path' effectively already has the target name. Skipping."
        continue
      fi

      if [[ -e "$new_file_path" ]]; then
        log_message "WARNING: Target '$new_file_path' already exists. Skipping rename of '$original_file_path'."
        echo "WARNING: Target '$new_file_path' already exists. Will not overwrite. Original: '$original_file_path'"
        continue
      fi

      if mv -n -- "$original_file_path" "$new_file_path"; then
        log_message "SUCCESS: Renamed '$original_file_path' to '$new_file_path'"
        renamed_in_dir_count=$((renamed_in_dir_count + 1))
      else
        local mv_exit_code=$?
        log_message "ERROR: Failed to rename '$original_file_path' to '$new_file_path'. mv exit code: $mv_exit_code"
        echo "ERROR: Failed to rename '$original_file_path'. Check log for details."
      fi
    done

    echo "Finished processing directory: $current_dir. Renamed $renamed_in_dir_count file(s) in this directory."
    if [[ $renamed_in_dir_count -gt 0 ]]; then
      log_message "Summary for $current_dir: Renamed $renamed_in_dir_count file(s)."
    fi
    echo # Blank line for readability
  done

  log_message "Script finished."
  echo "All operations complete. Check '$LOG_FILE' for detailed logs."
}

# Execute the main function with all script arguments
main "$@"