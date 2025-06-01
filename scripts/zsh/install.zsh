#!/usr/bin/zsh
# ================================================================================
# - File:    install.zsh
# - Purpose: Install/Update csalt++ template header files
#
# Source Metadata
# - Author:  Jonathan A. Webb
# - Date:    June 1, 2025
# - Version: 1.0
# - Copyright: Copyright 2025, Jon Webb Inc.
# ================================================================================

# Check for root
if [ "$EUID" -ne 0 ]; then
   echo "Please run as root (use sudo)"
   exit 1
fi

# OS detection
if [[ "$OSTYPE" == "darwin"* ]]; then
   INCLUDE_DIR="/usr/local/include/csaltpp"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
   INCLUDE_DIR="/usr/include/csaltpp"
else
   echo "Unsupported OS"
   exit 1
fi

# Create dirs
mkdir -p "$INCLUDE_DIR"
BACKUP_DIR="/tmp/csaltpp_backup"
mkdir -p "$BACKUP_DIR"

# Install header function
install_header() {
   local src=$1
   local dest=$2
   local desc=$3

   if [ -f "$dest" ]; then
       echo "Updating existing $desc..."
       local backup_path="$BACKUP_DIR/$(basename $dest).$(date +%Y%m%d_%H%M%S)"
       cp "$dest" "$backup_path"
       echo "Backed up to $backup_path"
   else
       echo "Installing new $desc..."
   fi

   cp "$src" "$dest"
   if [ $? -eq 0 ]; then
       echo "$desc installed successfully"
       chmod 644 "$dest"
   else
       echo "Error installing $desc"
       return 1
   fi
   return 0
}

# Install header-only templates from include/
echo -e "\nInstalling csalt++ header files..."
for file in ../../csalt++/include/*.hpp; do
    filename=$(basename "$file")
    install_header "$file" "$INCLUDE_DIR/$filename" "$filename" || exit 1
done

echo -e "\nInstallation/Update completed successfully"
echo "Backups (if any) stored in $BACKUP_DIR"
# ================================================================================
# eof

