#!/usr/bin/env bash
# ================================================================================
# - File:    install.sh
# - Purpose: Install/Update csalt++ template headers to system include directory
# ================================================================================

# Check for root
if [ "$EUID" -ne 0 ]; then
    printf "Please run as root (use sudo)\n"
    exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    INCLUDE_DIR="/usr/local/include/csaltpp"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    INCLUDE_DIR="/usr/include/csaltpp"
else
    printf "Unsupported operating system\n"
    exit 1
fi

# Create directories
mkdir -p "$INCLUDE_DIR"
BACKUP_DIR="/tmp/csaltpp_backup"
mkdir -p "$BACKUP_DIR"

# Function to backup and install a file
install_header() {
    local src=$1
    local dest=$2
    local desc=$3

    if [ -f "$dest" ]; then
        printf "Updating existing %s...\n" "$desc"
        local backup_path="$BACKUP_DIR/$(basename "$dest").$(date +%Y%m%d_%H%M%S)"
        cp "$dest" "$backup_path"
        printf "Backed up to %s\n" "$backup_path"
    else
        printf "Installing new %s...\n" "$desc"
    fi

    cp "$src" "$dest"
    if [ $? -eq 0 ]; then
        printf "%s installed successfully\n" "$desc"
        chmod 644 "$dest"
    else
        printf "Error installing %s\n" "$desc"
        return 1
    fi
    return 0
}

# Install header-only templates from include/
printf "\nInstalling csalt++ header files...\n"
for file in ../../csalt++/include/*.hpp; do
    filename=$(basename "$file")
    install_header "$file" "$INCLUDE_DIR/$filename" "$filename" || exit 1
done

printf "\nInstallation/Update completed successfully\n"
printf "Backups (if any) are stored in %s\n" "$BACKUP_DIR"
# ================================================================================
# ================================================================================ 
# eof

