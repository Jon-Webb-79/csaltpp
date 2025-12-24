// ================================================================================
// ================================================================================
// - File:    error.cpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    December 24, 2025
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#include "error.hpp"
// ================================================================================ 
// ================================================================================ 


namespace cslt {

    Error::Error() {
        safeCopy(message, "An error occurred", MAX_MESSAGE_LEN);
    }
// -------------------------------------------------------------------------------- 

    Error::Error(const char* msg) {
        safeCopy(message, msg, MAX_MESSAGE_LEN);
    }
// ================================================================================ 

    const char* Error::what() const {
        return message;
    }

    Error::~Error() {}

    void Error::safeCopy(char* dest, const char* src, size_t maxLen) {
        size_t i = 0;
        while (i < maxLen - 1 && src[i] != '\0') {
            dest[i] = src[i];
            i++;
        }
        dest[i] = '\0';
    }

    void Error::append(char* dest, const char* suffix, size_t maxLen) {
        size_t len = 0;
        while (dest[len] != '\0' && len < maxLen) len++;
        
        size_t i = 0;
        while (len < maxLen - 1 && suffix[i] != '\0') {
            dest[len++] = suffix[i++];
        }
        dest[len] = '\0';
    }
// -------------------------------------------------------------------------------- 

    void Error::prepend(char* dest, const char* prefix, size_t maxLen) {
        char temp[MAX_MESSAGE_LEN];
        safeCopy(temp, dest, MAX_MESSAGE_LEN);
        safeCopy(dest, prefix, maxLen);
        append(dest, temp, maxLen);
    }
// -------------------------------------------------------------------------------- 

    void Error::compose(char* dest, const char* prefix, const char* suffix, size_t maxLen) {
        safeCopy(dest, prefix, maxLen);
        append(dest, suffix, maxLen);
    }
// ================================================================================ 
// ================================================================================ 

    ArgumentError::ArgumentError() : Error("Invalid argument") {}
// -------------------------------------------------------------------------------- 

    ArgumentError::ArgumentError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    InvalidArgError::InvalidArgError() : ArgumentError("Invalid function argument") {}
// -------------------------------------------------------------------------------- 

    InvalidArgError::InvalidArgError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    NullPointerError::NullPointerError() : ArgumentError("Null pointer passed") {}
// -------------------------------------------------------------------------------- 

    NullPointerError::NullPointerError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    OutOfBoundsError::OutOfBoundsError() : ArgumentError("Index out of range") {}
// -------------------------------------------------------------------------------- 

    OutOfBoundsError::OutOfBoundsError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    SizeMismatchError::SizeMismatchError() : ArgumentError("Dimension/size mismatch") {}
// -------------------------------------------------------------------------------- 

    SizeMismatchError::SizeMismatchError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    UninitializedError::UninitializedError() : ArgumentError("Uninitialized element access") {}
// -------------------------------------------------------------------------------- 

    UninitializedError::UninitializedError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IteratorInvalidError::IteratorInvalidError() : ArgumentError("Invalid iterator/cursor") {}
// -------------------------------------------------------------------------------- 

    IteratorInvalidError::IteratorInvalidError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    PreconditionFailError::PreconditionFailError() : ArgumentError("Precondition failed") {}
// -------------------------------------------------------------------------------- 

    PreconditionFailError::PreconditionFailError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    PostconditionFailError::PostconditionFailError() : ArgumentError("Postcondition failed") {}
// -------------------------------------------------------------------------------- 

    PostconditionFailError::PostconditionFailError(const char* msg) : ArgumentError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IllegalStateError::IllegalStateError() : ArgumentError("Illegal state for operation") {}
// -------------------------------------------------------------------------------- 

    IllegalStateError::IllegalStateError(const char* msg) : ArgumentError(msg) {}
// ================================================================================
// ================================================================================ 

    MemoryError::MemoryError() : Error("Memory allocation failed") {}
// -------------------------------------------------------------------------------- 

    MemoryError::MemoryError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    StateError::StateError() : Error("Invalid state") {}
// -------------------------------------------------------------------------------- 

    StateError::StateError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    MathError::MathError() : Error("Mathematical error") {}
// -------------------------------------------------------------------------------- 

    MathError::MathError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOError::IOError() : Error("I/O operation failed") {}
// -------------------------------------------------------------------------------- 

    IOError::IOError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    FormatError::FormatError() : Error("Format error") {}
// -------------------------------------------------------------------------------- 

    FormatError::FormatError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    ConcurrencyError::ConcurrencyError() : Error("Concurrency error") {}
// -------------------------------------------------------------------------------- 

    ConcurrencyError::ConcurrencyError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    ConfigError::ConfigError() : Error("Configuration error") {}
// -------------------------------------------------------------------------------- 

    ConfigError::ConfigError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    GenericError::GenericError() : Error("An error occurred") {}
// -------------------------------------------------------------------------------- 

    GenericError::GenericError(const char* msg) : Error(msg) {}
} // namespace cslt
// ================================================================================
// ================================================================================
// eof
