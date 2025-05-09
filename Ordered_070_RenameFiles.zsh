#!/bin/zsh

# Script to recursively find files like 'prefix_nee_newname.pkl'
# and rename them to 'newname.pkl'.
# It logs actions and provides per-directory feedback.

# Ran with:
# /Users/kkreth/PycharmProjects/cgan/Ordered_070_RenameFiles.zsh /Users/kkreth/PycharmProjects/data/all_data_ready_for_training /tmp/rename.log


# Function to log messages with a timestamp
log_message() {
  # Ensure LOG_FILE is available in this function's scope if it's not passed
  # For this script structure, LOG_FILE is global to main, so accessible.
  echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

main() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <target_directory> [log_file_path]"
    echo "Example: $0 /path/to/your/files operations.log"
    return 1
  fi

  local TARGET_DIR
  TARGET_DIR="$(realpath "$1")" # Resolve to absolute path
  LOG_FILE="${2:-rename_files.log}" # Default log file name if not provided

  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory '$TARGET_DIR' not found or is not a directory."
    return 1
  fi

  # Attempt to create or touch the log file to check writability
  echo "Attempting to log to: $(realpath "$LOG_FILE" 2>/dev/null || echo "$LOG_FILE")"
  touch "$LOG_FILE" &>/dev/null
  if [[ $? -ne 0 ]]; then
    # Try to get directory of log file to check if it's writable
    local log_dir
    log_dir="$(dirname "$LOG_FILE")"
    if [[ ! -w "$log_dir" ]]; then
        echo "Error: Cannot write to log file '$LOG_FILE'. Directory '$log_dir' may not be writable or log file is not accessible."
        return 1
    fi
    # If directory is writable but touch failed, it might be a permission issue with the file itself.
    # For simplicity, we'll proceed, and actual write errors will be caught by redirection.
    echo "Warning: Could not touch log file '$LOG_FILE', but will attempt to write."
  fi

  log_message "Script started. Target directory: '$TARGET_DIR'"
  echo "Processing files in '$TARGET_DIR' and its subdirectories."
  echo "Logging operations to '$LOG_FILE'."
  echo # Blank line for readability

  # Find all directories under TARGET_DIR (including TARGET_DIR itself)
  # -print0 and read -d $'\0' are used for safe handling of names with spaces/special chars
  find "$TARGET_DIR" -type d -print0 | sort -z | while IFS= read -r -d $'\0' current_dir; do
    local renamed_in_dir_count=0
    echo "Scanning directory: $current_dir"

    # Find files matching the pattern directly within current_dir (maxdepth 1)
    # Pattern: anything_nee_thenTheRealName.pkl
    find "$current_dir" -maxdepth 1 -type f -name '*_nee_*.pkl' -print0 | while IFS= read -r -d $'\0' original_file_path; do
      local original_filename
      original_filename="${original_file_path##*/}" # Extract filename from path

      # Extract the part after '_nee_' which will be the new filename
      # e.g., if original_filename is '1_nee_135.0.pkl', new_basename becomes '135.0.pkl'
      # e.g., if original_filename is '2_nee_242.pkl', new_basename becomes '242.pkl'
      local new_basename
      new_basename="${original_filename#*_nee_}"

      if [[ -z "$new_basename" ]] || [[ "$new_basename" == "$original_filename" ]]; then
        # This might happen if filename is just "_nee_.pkl" or doesn't properly match
        log_message "SKIPPING: Could not derive new name for '$original_file_path' from '$original_filename'. Pattern mismatch?"
        continue
      fi

      local dir_of_file
      dir_of_file="${original_file_path%/*}"
      local new_file_path="${dir_of_file}/${new_basename}"

      # Check if renaming is actually needed (it should be, given the find pattern)
      if [[ "$original_file_path" == "$new_file_path" ]]; then
        log_message "INFO: File '$original_file_path' already has the target name. Skipping."
        continue
      fi

      # Check for potential overwrite
      if [[ -e "$new_file_path" ]]; then
        log_message "WARNING: Target '$new_file_path' already exists. Skipping rename of '$original_file_path'."
        echo "WARNING: Target '$new_file_path' already exists. Will not overwrite. Original: '$original_file_path'"
        continue
      fi

      # Perform the rename
      if mv -n -- "$original_file_path" "$new_file_path"; then # -n for no-clobber, though we check above
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