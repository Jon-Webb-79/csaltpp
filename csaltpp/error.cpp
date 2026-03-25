// ================================================================================
// ================================================================================
// - File:    error.cpp
// - Purpose: This file contains the implementation of error handling classes as 
//            part of the cslt namespace
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

    NoError::NoError() : Error("No Error") {}
// -------------------------------------------------------------------------------- 

    NoError::NoError(const char* msg) : Error(msg) {}
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

    BadAllocError::BadAllocError() : MemoryError("Memory allocation failed") {}
// -------------------------------------------------------------------------------- 

    BadAllocError::BadAllocError(const char* msg) : MemoryError(msg) {}
// ================================================================================ 
// ================================================================================ 

    ReallocFailError::ReallocFailError() : MemoryError("Memory reallocation failed") {}
// -------------------------------------------------------------------------------- 

    ReallocFailError::ReallocFailError(const char* msg) : MemoryError(msg) {}
// ================================================================================ 
// ================================================================================ 

    OutOfMemoryError::OutOfMemoryError() : MemoryError("Out of memory") {}
// -------------------------------------------------------------------------------- 

    OutOfMemoryError::OutOfMemoryError(const char* msg) : MemoryError(msg) {}
// ================================================================================ 
// ================================================================================ 

    LengthOverflowError::LengthOverflowError() : MemoryError("Length/size arithmetic overflow") {}
// -------------------------------------------------------------------------------- 

    LengthOverflowError::LengthOverflowError(const char* msg) : MemoryError(msg) {}
// ================================================================================ 
// ================================================================================ 

    CapacityOverflowError::CapacityOverflowError() : MemoryError("Capacity limit exceeded") {}
// -------------------------------------------------------------------------------- 

    CapacityOverflowError::CapacityOverflowError(const char* msg) : MemoryError(msg) {}
// ================================================================================ 
// ================================================================================ 

    AlignmentError::AlignmentError() : MemoryError("Required alignment not satisfied") {}
// -------------------------------------------------------------------------------- 

    AlignmentError::AlignmentError(const char* msg) : MemoryError(msg) {}

// ================================================================================ 
// ================================================================================ 

    StateError::StateError() : Error("Invalid state") {}
// -------------------------------------------------------------------------------- 

    StateError::StateError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    StateCorruptError::StateCorruptError() : StateError("Internal state corrupt") {}
// -------------------------------------------------------------------------------- 

    StateCorruptError::StateCorruptError(const char* msg) : StateError(msg) {}
// ================================================================================ 
// ================================================================================ 

    AlreadyInitializedError::AlreadyInitializedError() : StateError("Already initialized") {}
// -------------------------------------------------------------------------------- 

    AlreadyInitializedError::AlreadyInitializedError(const char* msg) : StateError(msg) {}
// ================================================================================ 
// ================================================================================ 

    NotFoundError::NotFoundError() : StateError("Item not found") {}
// -------------------------------------------------------------------------------- 

    NotFoundError::NotFoundError(const char* msg) : StateError(msg) {}
// ================================================================================ 
// ================================================================================ 

    EmptyError::EmptyError() : StateError("Container is empty") {}
// -------------------------------------------------------------------------------- 

    EmptyError::EmptyError(const char* msg) : StateError(msg) {}
// ================================================================================ 
// ================================================================================ 

    ConcurrentModificationError::ConcurrentModificationError() : StateError("Concurrent modification detected") {}
// -------------------------------------------------------------------------------- 

    ConcurrentModificationError::ConcurrentModificationError(const char* msg) : StateError(msg) {}
// ================================================================================ 
// ================================================================================ 

    MathError::MathError() : Error("Mathematical error") {}
// -------------------------------------------------------------------------------- 

    MathError::MathError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    DivByZeroError::DivByZeroError() : MathError("Division by zero") {}
// -------------------------------------------------------------------------------- 

    DivByZeroError::DivByZeroError(const char* msg) : MathError(msg) {}
// ================================================================================ 
// ================================================================================ 

    SingularMatrixError::SingularMatrixError() : MathError("Singular/non-invertible matrix") {}
// -------------------------------------------------------------------------------- 

    SingularMatrixError::SingularMatrixError(const char* msg) : MathError(msg) {}
// ================================================================================ 
// ================================================================================ 

    NumericOverflowError::NumericOverflowError() : MathError("Numeric overflow/underflow") {}
// -------------------------------------------------------------------------------- 

    NumericOverflowError::NumericOverflowError(const char* msg) : MathError(msg) {}
// ================================================================================ 
// ================================================================================ 

    DomainError::DomainError() : MathError("Math domain error") {}
// -------------------------------------------------------------------------------- 

    DomainError::DomainError(const char* msg) : MathError(msg) {}
// ================================================================================ 
// ================================================================================ 

    LossOfPrecisionError::LossOfPrecisionError() : MathError("Loss of numeric precision") {}
// -------------------------------------------------------------------------------- 

    LossOfPrecisionError::LossOfPrecisionError(const char* msg) : MathError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOError::IOError() : Error("I/O operation failed") {}
// -------------------------------------------------------------------------------- 

    IOError::IOError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    FileOpenError::FileOpenError() : IOError("Failed to open file/handle") {}
// -------------------------------------------------------------------------------- 

    FileOpenError::FileOpenError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    FileReadError::FileReadError() : IOError("Error reading from file/handle") {}
// -------------------------------------------------------------------------------- 

    FileReadError::FileReadError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    FileWriteError::FileWriteError() : IOError("Error writing to file/handle") {}
// -------------------------------------------------------------------------------- 

    FileWriteError::FileWriteError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    PermissionDeniedError::PermissionDeniedError() : IOError("Permission denied") {}
// -------------------------------------------------------------------------------- 

    PermissionDeniedError::PermissionDeniedError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOInterruptedError::IOInterruptedError() : IOError("I/O interrupted") {}
// -------------------------------------------------------------------------------- 

    IOInterruptedError::IOInterruptedError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOTimeoutError::IOTimeoutError() : IOError("I/O timed out") {}
// -------------------------------------------------------------------------------- 

    IOTimeoutError::IOTimeoutError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOClosedError::IOClosedError() : IOError("Operation on closed stream/descriptor") {}
// -------------------------------------------------------------------------------- 

    IOClosedError::IOClosedError(const char* msg) : IOError(msg) {}
// ================================================================================ 
// ================================================================================ 

    IOWouldBlockError::IOWouldBlockError() : IOError("Operation would block") {}
// -------------------------------------------------------------------------------- 

    IOWouldBlockError::IOWouldBlockError(const char* msg) : IOError(msg) {}

// ================================================================================ 
// ================================================================================ 

    FormatError::FormatError() : Error("Format error") {}
// -------------------------------------------------------------------------------- 

    FormatError::FormatError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    TypeMismatchError::TypeMismatchError()
        : FormatError("Type mismatch") {}
// --------------------------------------------------------------------------------

    TypeMismatchError::TypeMismatchError(const char* msg)
        : FormatError(msg) {}
// ================================================================================

    FormatInvalidError::FormatInvalidError()
        : FormatError("Invalid data format") {}
// --------------------------------------------------------------------------------

    FormatInvalidError::FormatInvalidError(const char* msg)
        : FormatError(msg) {}
// ================================================================================

    EncodingInvalidError::EncodingInvalidError()
        : FormatError("Invalid text encoding") {}
// --------------------------------------------------------------------------------

    EncodingInvalidError::EncodingInvalidError(const char* msg)
        : FormatError(msg) {}
// ================================================================================

    ParsingFailedError::ParsingFailedError()
        : FormatError("Parsing failed") {}
// --------------------------------------------------------------------------------

    ParsingFailedError::ParsingFailedError(const char* msg)
        : FormatError(msg) {}
// ================================================================================

    ValidationFailedError::ValidationFailedError()
        : FormatError("Validation failed") {}
// --------------------------------------------------------------------------------

    ValidationFailedError::ValidationFailedError(const char* msg)
        : FormatError(msg) {}
// ================================================================================
// ================================================================================

    ConcurrencyError::ConcurrencyError() : Error("Concurrency error") {}
// -------------------------------------------------------------------------------- 

    ConcurrencyError::ConcurrencyError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    LockFailedError::LockFailedError()
        : ConcurrencyError("Lock operation failed") {}
// --------------------------------------------------------------------------------

    LockFailedError::LockFailedError(const char* msg)
        : ConcurrencyError(msg) {}
// ================================================================================

    DeadlockDetectedError::DeadlockDetectedError()
        : ConcurrencyError("Deadlock detected") {}
// --------------------------------------------------------------------------------

    DeadlockDetectedError::DeadlockDetectedError(const char* msg)
        : ConcurrencyError(msg) {}
// ================================================================================

    ThreadFailError::ThreadFailError()
        : ConcurrencyError("Thread operation failed") {}
// --------------------------------------------------------------------------------

    ThreadFailError::ThreadFailError(const char* msg)
        : ConcurrencyError(msg) {}
// ================================================================================

    CancelledError::CancelledError()
        : ConcurrencyError("Operation cancelled") {}
// --------------------------------------------------------------------------------

    CancelledError::CancelledError(const char* msg)
        : ConcurrencyError(msg) {}
// ================================================================================

    RaceDetectedError::RaceDetectedError()
        : ConcurrencyError("Data race detected") {}
// --------------------------------------------------------------------------------

    RaceDetectedError::RaceDetectedError(const char* msg)
        : ConcurrencyError(msg) {}
// ================================================================================ 
// ================================================================================ 

    ConfigError::ConfigError() : Error("Configuration error") {}
// -------------------------------------------------------------------------------- 

    ConfigError::ConfigError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    ConfigInvalidError::ConfigInvalidError()
        : ConfigError("Invalid configuration") {}
// --------------------------------------------------------------------------------

    ConfigInvalidError::ConfigInvalidError(const char* msg)
        : ConfigError(msg) {}
// ================================================================================

    UnsupportedError::UnsupportedError()
        : ConfigError("Unsupported feature/platform") {}
// --------------------------------------------------------------------------------

    UnsupportedError::UnsupportedError(const char* msg)
        : ConfigError(msg) {}
// ================================================================================

    FeatureDisabledError::FeatureDisabledError()
        : ConfigError("Feature disabled by policy/build") {}
// --------------------------------------------------------------------------------

    FeatureDisabledError::FeatureDisabledError(const char* msg)
        : ConfigError(msg) {}
// ================================================================================

    VersionMismatchError::VersionMismatchError()
        : ConfigError("Version/ABI mismatch") {}
// --------------------------------------------------------------------------------

    VersionMismatchError::VersionMismatchError(const char* msg)
        : ConfigError(msg) {}
// ================================================================================

    ResourceExhaustedError::ResourceExhaustedError()
        : ConfigError("Resource exhausted") {}
// --------------------------------------------------------------------------------

    ResourceExhaustedError::ResourceExhaustedError(const char* msg)
        : ConfigError(msg) {}

// ================================================================================ 
// ================================================================================ 

    GenericError::GenericError() : Error("An error occurred") {}
// -------------------------------------------------------------------------------- 

    GenericError::GenericError(const char* msg) : Error(msg) {}
// ================================================================================ 
// ================================================================================ 

    NotImplementedError::NotImplementedError()
        : GenericError("Not implemented") {}
// --------------------------------------------------------------------------------

    NotImplementedError::NotImplementedError(const char* msg)
        : GenericError(msg) {}
// ================================================================================

    OperationUnavailableError::OperationUnavailableError()
        : GenericError("Operation unavailable") {}
// --------------------------------------------------------------------------------

    OperationUnavailableError::OperationUnavailableError(const char* msg)
        : GenericError(msg) {}
// ================================================================================

    UnknownError::UnknownError()
        : GenericError("Unknown error") {}
// --------------------------------------------------------------------------------

    UnknownError::UnknownError(const char* msg)
        : GenericError(msg) {}
// ================================================================================
// ================================================================================

} // namespace cslt
// ================================================================================
// ================================================================================
// eof
