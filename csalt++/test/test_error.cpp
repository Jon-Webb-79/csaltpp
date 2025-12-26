// ================================================================================
// ================================================================================
// - File:    test_error.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    December 24, 2025
// - Version: 1.0
// - Copyright: Copyright 2025, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>
#include <climits>
#include "error.hpp"

using namespace cslt;
// ================================================================================ 
// ================================================================================ 
// Test Error Class 

class ErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};
// -------------------------------------------------------------------------------- 

// Test default constructor uses predefined message
TEST_F(ErrorTest, DefaultConstructorUsesPredefinedMessage) {
    Error err;
    EXPECT_STREQ("An error occurred", err.what());
}
// -------------------------------------------------------------------------------- 

// Test custom message constructor
TEST_F(ErrorTest, CustomMessageConstructor) {
    Error err("Custom error message");
    EXPECT_STREQ("Custom error message", err.what());
}
// -------------------------------------------------------------------------------- 

// Test that what() returns non-null pointer
TEST_F(ErrorTest, WhatReturnsNonNull) {
    Error err;
    EXPECT_NE(nullptr, err.what());
}
// -------------------------------------------------------------------------------- 

// Test short message
TEST_F(ErrorTest, ShortMessage) {
    Error err("Short");
    EXPECT_STREQ("Short", err.what());
}
// -------------------------------------------------------------------------------- 

// Test empty message
TEST_F(ErrorTest, EmptyMessage) {
    Error err("");
    EXPECT_STREQ("", err.what());
}
// -------------------------------------------------------------------------------- 

// Test message at maximum length (255 chars + null terminator)
TEST_F(ErrorTest, MaxLengthMessage) {
    // Create a string of exactly 255 characters
    char maxMsg[256];
    for (int i = 0; i < 255; i++) {
        maxMsg[i] = 'A';
    }
    maxMsg[255] = '\0';
    
    Error err(maxMsg);
    EXPECT_STREQ(maxMsg, err.what());
    EXPECT_EQ(255, strlen(err.what()));
}
// -------------------------------------------------------------------------------- 

// Test message truncation (message longer than MAX_MESSAGE_LEN)
TEST_F(ErrorTest, MessageTruncation) {
    // Create a string of 300 characters (exceeds 256 limit)
    char longMsg[301];
    for (int i = 0; i < 300; i++) {
        longMsg[i] = 'B';
    }
    longMsg[300] = '\0';
    
    Error err(longMsg);
    
    // Message should be truncated to 255 characters
    EXPECT_EQ(255, strlen(err.what()));
    
    // First 255 characters should match
    char expected[256];
    for (int i = 0; i < 255; i++) {
        expected[i] = 'B';
    }
    expected[255] = '\0';
    EXPECT_STREQ(expected, err.what());
}
// -------------------------------------------------------------------------------- 

// Test throwing and catching with default message
TEST_F(ErrorTest, ThrowAndCatchDefaultMessage) {
    try {
        throw Error();
    } catch (const Error& e) {
        EXPECT_STREQ("An error occurred", e.what());
    }
}
// -------------------------------------------------------------------------------- 

// Test throwing and catching with custom message
TEST_F(ErrorTest, ThrowAndCatchCustomMessage) {
    try {
        throw Error("Something went wrong");
    } catch (const Error& e) {
        EXPECT_STREQ("Something went wrong", e.what());
    }
}
// -------------------------------------------------------------------------------- 

// Test copy constructor (if applicable)
TEST_F(ErrorTest, CopyConstructor) {
    Error err1("Original message");
    Error err2(err1);
    EXPECT_STREQ("Original message", err2.what());
}
// -------------------------------------------------------------------------------- 

// Test that multiple Error objects are independent
TEST_F(ErrorTest, MultipleErrorsAreIndependent) {
    Error err1("First error");
    Error err2("Second error");
    
    EXPECT_STREQ("First error", err1.what());
    EXPECT_STREQ("Second error", err2.what());
}
// -------------------------------------------------------------------------------- 

// Test message with special characters
TEST_F(ErrorTest, MessageWithSpecialCharacters) {
    Error err("Error: File not found!\n\tPath: /home/user");
    EXPECT_STREQ("Error: File not found!\n\tPath: /home/user", err.what());
}
// -------------------------------------------------------------------------------- 

// Test message with numbers
TEST_F(ErrorTest, MessageWithNumbers) {
    Error err("Error code: 404");
    EXPECT_STREQ("Error code: 404", err.what());
}
// ================================================================================ 
// ================================================================================ 

TEST(ArgumentErrorTest, DefaultConstructor) {
    ArgumentError err;
    EXPECT_STREQ("Invalid argument", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ArgumentErrorTest, CustomMessage) {
    ArgumentError err("Index out of bounds");
    EXPECT_STREQ("Index out of bounds", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ArgumentErrorTest, ThrowAndCatch) {
    try {
        throw ArgumentError("Null pointer passed");
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Null pointer passed", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch ArgumentError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(ArgumentErrorTest, CatchAsBaseClass) {
    try {
        throw ArgumentError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid argument", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(MemoryErrorTest, DefaultConstructor) {
    MemoryError err;
    EXPECT_STREQ("Memory allocation failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorTest, CustomMessage) {
    MemoryError err("Failed to allocate 1024 bytes");
    EXPECT_STREQ("Failed to allocate 1024 bytes", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorTest, ThrowAndCatch) {
    try {
        throw MemoryError("Out of memory");
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Out of memory", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch MemoryError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorTest, CatchAsBaseClass) {
    try {
        throw MemoryError();
    } catch (const Error& e) {
        EXPECT_STREQ("Memory allocation failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(StateErrorTest, DefaultConstructor) {
    StateError err;
    EXPECT_STREQ("Invalid state", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorTest, CustomMessage) {
    StateError err("Container already initialized");
    EXPECT_STREQ("Container already initialized", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorTest, ThrowAndCatch) {
    try {
        throw StateError("Corrupt state detected");
    } catch (const StateError& e) {
        EXPECT_STREQ("Corrupt state detected", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch StateError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorTest, CatchAsBaseClass) {
    try {
        throw StateError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid state", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(MathErrorTest, DefaultConstructor) {
    MathError err;
    EXPECT_STREQ("Mathematical error", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorTest, CustomMessage) {
    MathError err("Division by zero");
    EXPECT_STREQ("Division by zero", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorTest, ThrowAndCatch) {
    try {
        throw MathError("Singular matrix");
    } catch (const MathError& e) {
        EXPECT_STREQ("Singular matrix", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch MathError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorTest, CatchAsBaseClass) {
    try {
        throw MathError();
    } catch (const Error& e) {
        EXPECT_STREQ("Mathematical error", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(IOErrorTest, DefaultConstructor) {
    IOError err;
    EXPECT_STREQ("I/O operation failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorTest, CustomMessage) {
    IOError err("Failed to open file");
    EXPECT_STREQ("Failed to open file", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorTest, ThrowAndCatch) {
    try {
        throw IOError("Permission denied");
    } catch (const IOError& e) {
        EXPECT_STREQ("Permission denied", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch IOError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorTest, CatchAsBaseClass) {
    try {
        throw IOError();
    } catch (const Error& e) {
        EXPECT_STREQ("I/O operation failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(FormatErrorTest, DefaultConstructor) {
    FormatError err;
    EXPECT_STREQ("Format error", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FormatErrorTest, CustomMessage) {
    FormatError err("Invalid JSON format");
    EXPECT_STREQ("Invalid JSON format", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FormatErrorTest, ThrowAndCatch) {
    try {
        throw FormatError("Type mismatch");
    } catch (const FormatError& e) {
        EXPECT_STREQ("Type mismatch", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch FormatError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(FormatErrorTest, CatchAsBaseClass) {
    try {
        throw FormatError();
    } catch (const Error& e) {
        EXPECT_STREQ("Format error", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ConcurrencyErrorTest, DefaultConstructor) {
    ConcurrencyError err;
    EXPECT_STREQ("Concurrency error", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrencyErrorTest, CustomMessage) {
    ConcurrencyError err("Deadlock detected");
    EXPECT_STREQ("Deadlock detected", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrencyErrorTest, ThrowAndCatch) {
    try {
        throw ConcurrencyError("Lock acquisition failed");
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Lock acquisition failed", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch ConcurrencyError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrencyErrorTest, CatchAsBaseClass) {
    try {
        throw ConcurrencyError();
    } catch (const Error& e) {
        EXPECT_STREQ("Concurrency error", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ConfigErrorTest, DefaultConstructor) {
    ConfigError err;
    EXPECT_STREQ("Configuration error", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConfigErrorTest, CustomMessage) {
    ConfigError err("Unsupported platform");
    EXPECT_STREQ("Unsupported platform", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConfigErrorTest, ThrowAndCatch) {
    try {
        throw ConfigError("Version mismatch");
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Version mismatch", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch ConfigError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(ConfigErrorTest, CatchAsBaseClass) {
    try {
        throw ConfigError();
    } catch (const Error& e) {
        EXPECT_STREQ("Configuration error", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(GenericErrorTest, DefaultConstructor) {
    GenericError err;
    EXPECT_STREQ("An error occurred", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(GenericErrorTest, CustomMessage) {
    GenericError err("Not implemented");
    EXPECT_STREQ("Not implemented", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(GenericErrorTest, ThrowAndCatch) {
    try {
        throw GenericError("Unknown error");
    } catch (const GenericError& e) {
        EXPECT_STREQ("Unknown error", e.what());
    } catch (const Error& e) {
        FAIL() << "Should catch GenericError, not base Error";
    }
}
// -------------------------------------------------------------------------------- 

TEST(GenericErrorTest, CatchAsBaseClass) {
    try {
        throw GenericError();
    } catch (const Error& e) {
        EXPECT_STREQ("An error occurred", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ErrorHierarchyTest, CatchSpecificTypesInOrder) {
    bool caught_memory_error = false;
    bool caught_io_error = false;
    bool caught_base_error = false;
    
    try {
        throw MemoryError("Out of memory");
    } catch (const MemoryError& e) {
        caught_memory_error = true;
    } catch (const Error& e) {
        caught_base_error = true;
    }
    
    EXPECT_TRUE(caught_memory_error);
    EXPECT_FALSE(caught_base_error);
    
    try {
        throw IOError("File not found");
    } catch (const IOError& e) {
        caught_io_error = true;
    } catch (const Error& e) {
        caught_base_error = true;
    }
    
    EXPECT_TRUE(caught_io_error);
    EXPECT_FALSE(caught_base_error);
}
// -------------------------------------------------------------------------------- 

TEST(ErrorHierarchyTest, CatchMultipleErrorTypes) {
    int argument_count = 0;
    int memory_count = 0;
    int other_count = 0;
    
    for (int i = 0; i < 3; i++) {
        try {
            if (i == 0) throw ArgumentError("Bad arg");
            if (i == 1) throw MemoryError("Bad alloc");
            if (i == 2) throw MathError("Bad math");
        } catch (const ArgumentError& e) {
            argument_count++;
        } catch (const MemoryError& e) {
            memory_count++;
        } catch (const Error& e) {
            other_count++;
        }
    }
    
    EXPECT_EQ(1, argument_count);
    EXPECT_EQ(1, memory_count);
    EXPECT_EQ(1, other_count);
}
// ================================================================================ 
// ================================================================================ 

TEST(InvalidArgErrorTest, DefaultConstructor) {
    InvalidArgError err;
    EXPECT_STREQ("Invalid function argument", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(InvalidArgErrorTest, CustomMessage) {
    InvalidArgError err("Expected positive value, got -5");
    EXPECT_STREQ("Expected positive value, got -5", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(InvalidArgErrorTest, ThrowAndCatchSpecific) {
    try {
        throw InvalidArgError("Bad parameter");
    } catch (const InvalidArgError& e) {
        EXPECT_STREQ("Bad parameter", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(InvalidArgErrorTest, CatchAsArgumentError) {
    try {
        throw InvalidArgError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Invalid function argument", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(InvalidArgErrorTest, CatchAsBaseError) {
    try {
        throw InvalidArgError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid function argument", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(NullPointerErrorTest, DefaultConstructor) {
    NullPointerError err;
    EXPECT_STREQ("Null pointer passed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NullPointerErrorTest, CustomMessage) {
    NullPointerError err("Buffer pointer is null");
    EXPECT_STREQ("Buffer pointer is null", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NullPointerErrorTest, ThrowAndCatchSpecific) {
    try {
        throw NullPointerError("Data pointer is null");
    } catch (const NullPointerError& e) {
        EXPECT_STREQ("Data pointer is null", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NullPointerErrorTest, CatchAsArgumentError) {
    try {
        throw NullPointerError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Null pointer passed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NullPointerErrorTest, CatchAsBaseError) {
    try {
        throw NullPointerError();
    } catch (const Error& e) {
        EXPECT_STREQ("Null pointer passed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(OutOfBoundsErrorTest, DefaultConstructor) {
    OutOfBoundsError err;
    EXPECT_STREQ("Index out of range", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(OutOfBoundsErrorTest, CustomMessage) {
    OutOfBoundsError err("Index 10 exceeds array size 5");
    EXPECT_STREQ("Index 10 exceeds array size 5", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(OutOfBoundsErrorTest, ThrowAndCatchSpecific) {
    try {
        throw OutOfBoundsError("Access beyond bounds");
    } catch (const OutOfBoundsError& e) {
        EXPECT_STREQ("Access beyond bounds", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(OutOfBoundsErrorTest, CatchAsArgumentError) {
    try {
        throw OutOfBoundsError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Index out of range", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(OutOfBoundsErrorTest, CatchAsBaseError) {
    try {
        throw OutOfBoundsError();
    } catch (const Error& e) {
        EXPECT_STREQ("Index out of range", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(SizeMismatchErrorTest, DefaultConstructor) {
    SizeMismatchError err;
    EXPECT_STREQ("Dimension/size mismatch", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(SizeMismatchErrorTest, CustomMessage) {
    SizeMismatchError err("Matrix dimensions incompatible: 3x4 and 2x3");
    EXPECT_STREQ("Matrix dimensions incompatible: 3x4 and 2x3", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(SizeMismatchErrorTest, ThrowAndCatchSpecific) {
    try {
        throw SizeMismatchError("Vector sizes don't match");
    } catch (const SizeMismatchError& e) {
        EXPECT_STREQ("Vector sizes don't match", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(SizeMismatchErrorTest, CatchAsArgumentError) {
    try {
        throw SizeMismatchError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Dimension/size mismatch", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(SizeMismatchErrorTest, CatchAsBaseError) {
    try {
        throw SizeMismatchError();
    } catch (const Error& e) {
        EXPECT_STREQ("Dimension/size mismatch", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(UninitializedErrorTest, DefaultConstructor) {
    UninitializedError err;
    EXPECT_STREQ("Uninitialized element access", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(UninitializedErrorTest, CustomMessage) {
    UninitializedError err("Attempting to read uninitialized variable");
    EXPECT_STREQ("Attempting to read uninitialized variable", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(UninitializedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw UninitializedError("Object not initialized");
    } catch (const UninitializedError& e) {
        EXPECT_STREQ("Object not initialized", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(UninitializedErrorTest, CatchAsArgumentError) {
    try {
        throw UninitializedError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Uninitialized element access", e.what());
    }
}

TEST(UninitializedErrorTest, CatchAsBaseError) {
    try {
        throw UninitializedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Uninitialized element access", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(IteratorInvalidErrorTest, DefaultConstructor) {
    IteratorInvalidError err;
    EXPECT_STREQ("Invalid iterator/cursor", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IteratorInvalidErrorTest, CustomMessage) {
    IteratorInvalidError err("Iterator invalidated by container modification");
    EXPECT_STREQ("Iterator invalidated by container modification", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IteratorInvalidErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IteratorInvalidError("Dangling iterator");
    } catch (const IteratorInvalidError& e) {
        EXPECT_STREQ("Dangling iterator", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IteratorInvalidErrorTest, CatchAsArgumentError) {
    try {
        throw IteratorInvalidError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Invalid iterator/cursor", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IteratorInvalidErrorTest, CatchAsBaseError) {
    try {
        throw IteratorInvalidError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid iterator/cursor", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(PreconditionFailErrorTest, DefaultConstructor) {
    PreconditionFailError err;
    EXPECT_STREQ("Precondition failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PreconditionFailErrorTest, CustomMessage) {
    PreconditionFailError err("Array must be sorted before binary search");
    EXPECT_STREQ("Array must be sorted before binary search", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PreconditionFailErrorTest, ThrowAndCatchSpecific) {
    try {
        throw PreconditionFailError("Input validation failed");
    } catch (const PreconditionFailError& e) {
        EXPECT_STREQ("Input validation failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PreconditionFailErrorTest, CatchAsArgumentError) {
    try {
        throw PreconditionFailError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Precondition failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PreconditionFailErrorTest, CatchAsBaseError) {
    try {
        throw PreconditionFailError();
    } catch (const Error& e) {
        EXPECT_STREQ("Precondition failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(PostconditionFailErrorTest, DefaultConstructor) {
    PostconditionFailError err;
    EXPECT_STREQ("Postcondition failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PostconditionFailErrorTest, CustomMessage) {
    PostconditionFailError err("Invariant violated: size must equal capacity");
    EXPECT_STREQ("Invariant violated: size must equal capacity", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PostconditionFailErrorTest, ThrowAndCatchSpecific) {
    try {
        throw PostconditionFailError("Result invariant broken");
    } catch (const PostconditionFailError& e) {
        EXPECT_STREQ("Result invariant broken", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PostconditionFailErrorTest, CatchAsArgumentError) {
    try {
        throw PostconditionFailError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Postcondition failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PostconditionFailErrorTest, CatchAsBaseError) {
    try {
        throw PostconditionFailError();
    } catch (const Error& e) {
        EXPECT_STREQ("Postcondition failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(IllegalStateErrorTest, DefaultConstructor) {
    IllegalStateError err;
    EXPECT_STREQ("Illegal state for operation", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IllegalStateErrorTest, CustomMessage) {
    IllegalStateError err("Cannot read from closed stream");
    EXPECT_STREQ("Cannot read from closed stream", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IllegalStateErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IllegalStateError("Operation not allowed in current state");
    } catch (const IllegalStateError& e) {
        EXPECT_STREQ("Operation not allowed in current state", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IllegalStateErrorTest, CatchAsArgumentError) {
    try {
        throw IllegalStateError();
    } catch (const ArgumentError& e) {
        EXPECT_STREQ("Illegal state for operation", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IllegalStateErrorTest, CatchAsBaseError) {
    try {
        throw IllegalStateError();
    } catch (const Error& e) {
        EXPECT_STREQ("Illegal state for operation", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ArgumentErrorHierarchyTest, CatchMultipleArgumentErrorTypes) {
    int null_ptr_count = 0;
    int out_of_bounds_count = 0;
    int other_arg_count = 0;
    
    for (int i = 0; i < 4; i++) {
        try {
            if (i == 0) throw NullPointerError();
            if (i == 1) throw OutOfBoundsError();
            if (i == 2) throw InvalidArgError();
            if (i == 3) throw IllegalStateError();
        } catch (const NullPointerError& e) {
            null_ptr_count++;
        } catch (const OutOfBoundsError& e) {
            out_of_bounds_count++;
        } catch (const ArgumentError& e) {
            other_arg_count++;
        }
    }
    
    EXPECT_EQ(1, null_ptr_count);
    EXPECT_EQ(1, out_of_bounds_count);
    EXPECT_EQ(2, other_arg_count);
}
// -------------------------------------------------------------------------------- 

TEST(ArgumentErrorHierarchyTest, AllArgumentErrorsCatchableAsBase) {
    bool all_caught = true;
    
    try {
        throw InvalidArgError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw NullPointerError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw OutOfBoundsError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw SizeMismatchError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw UninitializedError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IteratorInvalidError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw PreconditionFailError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw PostconditionFailError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IllegalStateError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    EXPECT_TRUE(all_caught);
}
// ================================================================================ 
// ================================================================================ 

TEST(BadAllocErrorTest, DefaultConstructor) {
    BadAllocError err;
    EXPECT_STREQ("Memory allocation failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(BadAllocErrorTest, CustomMessage) {
    BadAllocError err("Failed to allocate 1024 bytes for buffer");
    EXPECT_STREQ("Failed to allocate 1024 bytes for buffer", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(BadAllocErrorTest, ThrowAndCatchSpecific) {
    try {
        throw BadAllocError("malloc returned NULL");
    } catch (const BadAllocError& e) {
        EXPECT_STREQ("malloc returned NULL", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(BadAllocErrorTest, CatchAsMemoryError) {
    try {
        throw BadAllocError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Memory allocation failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(BadAllocErrorTest, CatchAsBaseError) {
    try {
        throw BadAllocError();
    } catch (const Error& e) {
        EXPECT_STREQ("Memory allocation failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ReallocFailErrorTest, DefaultConstructor) {
    ReallocFailError err;
    EXPECT_STREQ("Memory reallocation failed", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ReallocFailErrorTest, CustomMessage) {
    ReallocFailError err("Failed to expand buffer from 512 to 1024 bytes");
    EXPECT_STREQ("Failed to expand buffer from 512 to 1024 bytes", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ReallocFailErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ReallocFailError("realloc returned NULL");
    } catch (const ReallocFailError& e) {
        EXPECT_STREQ("realloc returned NULL", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(ReallocFailErrorTest, CatchAsMemoryError) {
    try {
        throw ReallocFailError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Memory reallocation failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(ReallocFailErrorTest, CatchAsBaseError) {
    try {
        throw ReallocFailError();
    } catch (const Error& e) {
        EXPECT_STREQ("Memory reallocation failed", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(OutOfMemoryErrorTest, DefaultConstructor) {
    OutOfMemoryError err;
    EXPECT_STREQ("Out of memory", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(OutOfMemoryErrorTest, CustomMessage) {
    OutOfMemoryError err("System memory exhausted");
    EXPECT_STREQ("System memory exhausted", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(OutOfMemoryErrorTest, ThrowAndCatchSpecific) {
    try {
        throw OutOfMemoryError("Allocator limit reached");
    } catch (const OutOfMemoryError& e) {
        EXPECT_STREQ("Allocator limit reached", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(OutOfMemoryErrorTest, CatchAsMemoryError) {
    try {
        throw OutOfMemoryError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Out of memory", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(OutOfMemoryErrorTest, CatchAsBaseError) {
    try {
        throw OutOfMemoryError();
    } catch (const Error& e) {
        EXPECT_STREQ("Out of memory", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(LengthOverflowErrorTest, DefaultConstructor) {
    LengthOverflowError err;
    EXPECT_STREQ("Length/size arithmetic overflow", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(LengthOverflowErrorTest, CustomMessage) {
    LengthOverflowError err("Size calculation overflowed: SIZE_MAX exceeded");
    EXPECT_STREQ("Size calculation overflowed: SIZE_MAX exceeded", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(LengthOverflowErrorTest, ThrowAndCatchSpecific) {
    try {
        throw LengthOverflowError("Length computation overflow");
    } catch (const LengthOverflowError& e) {
        EXPECT_STREQ("Length computation overflow", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(LengthOverflowErrorTest, CatchAsMemoryError) {
    try {
        throw LengthOverflowError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Length/size arithmetic overflow", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(LengthOverflowErrorTest, CatchAsBaseError) {
    try {
        throw LengthOverflowError();
    } catch (const Error& e) {
        EXPECT_STREQ("Length/size arithmetic overflow", e.what());
    }
}
// ================================================================================ 
// ================================================================================ 

TEST(CapacityOverflowErrorTest, DefaultConstructor) {
    CapacityOverflowError err;
    EXPECT_STREQ("Capacity limit exceeded", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(CapacityOverflowErrorTest, CustomMessage) {
    CapacityOverflowError err("Container capacity cannot exceed 65535 elements");
    EXPECT_STREQ("Container capacity cannot exceed 65535 elements", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(CapacityOverflowErrorTest, ThrowAndCatchSpecific) {
    try {
        throw CapacityOverflowError("Max capacity reached");
    } catch (const CapacityOverflowError& e) {
        EXPECT_STREQ("Max capacity reached", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(CapacityOverflowErrorTest, CatchAsMemoryError) {
    try {
        throw CapacityOverflowError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Capacity limit exceeded", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(CapacityOverflowErrorTest, CatchAsBaseError) {
    try {
        throw CapacityOverflowError();
    } catch (const Error& e) {
        EXPECT_STREQ("Capacity limit exceeded", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(AlignmentErrorTest, DefaultConstructor) {
    AlignmentError err;
    EXPECT_STREQ("Required alignment not satisfied", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(AlignmentErrorTest, CustomMessage) {
    AlignmentError err("Pointer must be 16-byte aligned for SIMD operations");
    EXPECT_STREQ("Pointer must be 16-byte aligned for SIMD operations", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(AlignmentErrorTest, ThrowAndCatchSpecific) {
    try {
        throw AlignmentError("Misaligned memory access");
    } catch (const AlignmentError& e) {
        EXPECT_STREQ("Misaligned memory access", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(AlignmentErrorTest, CatchAsMemoryError) {
    try {
        throw AlignmentError();
    } catch (const MemoryError& e) {
        EXPECT_STREQ("Required alignment not satisfied", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(AlignmentErrorTest, CatchAsBaseError) {
    try {
        throw AlignmentError();
    } catch (const Error& e) {
        EXPECT_STREQ("Required alignment not satisfied", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(MemoryErrorHierarchyTest, CatchMultipleMemoryErrorTypes) {
    int bad_alloc_count = 0;
    int out_of_memory_count = 0;
    int other_memory_count = 0;
    
    for (int i = 0; i < 5; i++) {
        try {
            if (i == 0) throw BadAllocError();
            if (i == 1) throw OutOfMemoryError();
            if (i == 2) throw ReallocFailError();
            if (i == 3) throw LengthOverflowError();
            if (i == 4) throw AlignmentError();
        } catch (const BadAllocError& e) {
            bad_alloc_count++;
        } catch (const OutOfMemoryError& e) {
            out_of_memory_count++;
        } catch (const MemoryError& e) {
            other_memory_count++;
        }
    }
    
    EXPECT_EQ(1, bad_alloc_count);
    EXPECT_EQ(1, out_of_memory_count);
    EXPECT_EQ(3, other_memory_count);
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorHierarchyTest, AllMemoryErrorsCatchableAsBase) {
    bool all_caught = true;
    
    try {
        throw BadAllocError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw ReallocFailError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw OutOfMemoryError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw LengthOverflowError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw CapacityOverflowError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw AlignmentError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    EXPECT_TRUE(all_caught);
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorHierarchyTest, DifferentiateMemoryFromArgumentErrors) {
    bool caught_correctly = true;
    
    // Should catch as MemoryError
    try {
        throw BadAllocError();
    } catch (const MemoryError&) {
        // Success
    } catch (const ArgumentError&) {
        caught_correctly = false;
    }
    
    // Should NOT catch ArgumentError as MemoryError
    try {
        throw NullPointerError();
    } catch (const MemoryError&) {
        caught_correctly = false;
    } catch (const ArgumentError&) {
        // Success
    }
    
    EXPECT_TRUE(caught_correctly);
}
// -------------------------------------------------------------------------------- 

TEST(MemoryErrorHierarchyTest, CatchSpecificBeforeGeneral) {
    std::string caught_type;
    
    try {
        throw BadAllocError("specific error");
    } catch (const BadAllocError& e) {
        caught_type = "BadAllocError";
    } catch (const MemoryError& e) {
        caught_type = "MemoryError";
    } catch (const Error& e) {
        caught_type = "Error";
    }
    
    EXPECT_EQ("BadAllocError", caught_type);
}
// ================================================================================ 
// ================================================================================ 

TEST(StateCorruptErrorTest, DefaultConstructor) {
    StateCorruptError err;
    EXPECT_STREQ("Internal state corrupt", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(StateCorruptErrorTest, CustomMessage) {
    StateCorruptError err("Checksum mismatch: data corruption detected");
    EXPECT_STREQ("Checksum mismatch: data corruption detected", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(StateCorruptErrorTest, ThrowAndCatchSpecific) {
    try {
        throw StateCorruptError("Invariant violated");
    } catch (const StateCorruptError& e) {
        EXPECT_STREQ("Invariant violated", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(StateCorruptErrorTest, CatchAsStateError) {
    try {
        throw StateCorruptError();
    } catch (const StateError& e) {
        EXPECT_STREQ("Internal state corrupt", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(StateCorruptErrorTest, CatchAsBaseError) {
    try {
        throw StateCorruptError();
    } catch (const Error& e) {
        EXPECT_STREQ("Internal state corrupt", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(AlreadyInitializedErrorTest, DefaultConstructor) {
    AlreadyInitializedError err;
    EXPECT_STREQ("Already initialized", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(AlreadyInitializedErrorTest, CustomMessage) {
    AlreadyInitializedError err("Cannot reinitialize active connection");
    EXPECT_STREQ("Cannot reinitialize active connection", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(AlreadyInitializedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw AlreadyInitializedError("Double initialization detected");
    } catch (const AlreadyInitializedError& e) {
        EXPECT_STREQ("Double initialization detected", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(AlreadyInitializedErrorTest, CatchAsStateError) {
    try {
        throw AlreadyInitializedError();
    } catch (const StateError& e) {
        EXPECT_STREQ("Already initialized", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(AlreadyInitializedErrorTest, CatchAsBaseError) {
    try {
        throw AlreadyInitializedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Already initialized", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(NotFoundErrorTest, DefaultConstructor) {
    NotFoundError err;
    EXPECT_STREQ("Item not found", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NotFoundErrorTest, CustomMessage) {
    NotFoundError err("Key 'username' not found in dictionary");
    EXPECT_STREQ("Key 'username' not found in dictionary", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NotFoundErrorTest, ThrowAndCatchSpecific) {
    try {
        throw NotFoundError("Element does not exist");
    } catch (const NotFoundError& e) {
        EXPECT_STREQ("Element does not exist", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NotFoundErrorTest, CatchAsStateError) {
    try {
        throw NotFoundError();
    } catch (const StateError& e) {
        EXPECT_STREQ("Item not found", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NotFoundErrorTest, CatchAsBaseError) {
    try {
        throw NotFoundError();
    } catch (const Error& e) {
        EXPECT_STREQ("Item not found", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(EmptyErrorTest, DefaultConstructor) {
    EmptyError err;
    EXPECT_STREQ("Container is empty", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(EmptyErrorTest, CustomMessage) {
    EmptyError err("Cannot pop from empty stack");
    EXPECT_STREQ("Cannot pop from empty stack", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(EmptyErrorTest, ThrowAndCatchSpecific) {
    try {
        throw EmptyError("Queue is empty");
    } catch (const EmptyError& e) {
        EXPECT_STREQ("Queue is empty", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(EmptyErrorTest, CatchAsStateError) {
    try {
        throw EmptyError();
    } catch (const StateError& e) {
        EXPECT_STREQ("Container is empty", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(EmptyErrorTest, CatchAsBaseError) {
    try {
        throw EmptyError();
    } catch (const Error& e) {
        EXPECT_STREQ("Container is empty", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(ConcurrentModificationErrorTest, DefaultConstructor) {
    ConcurrentModificationError err;
    EXPECT_STREQ("Concurrent modification detected", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrentModificationErrorTest, CustomMessage) {
    ConcurrentModificationError err("Container modified during iteration");
    EXPECT_STREQ("Container modified during iteration", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrentModificationErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ConcurrentModificationError("Modification version mismatch");
    } catch (const ConcurrentModificationError& e) {
        EXPECT_STREQ("Modification version mismatch", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrentModificationErrorTest, CatchAsStateError) {
    try {
        throw ConcurrentModificationError();
    } catch (const StateError& e) {
        EXPECT_STREQ("Concurrent modification detected", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(ConcurrentModificationErrorTest, CatchAsBaseError) {
    try {
        throw ConcurrentModificationError();
    } catch (const Error& e) {
        EXPECT_STREQ("Concurrent modification detected", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(StateErrorHierarchyTest, CatchMultipleStateErrorTypes) {
    int corrupt_count = 0;
    int not_found_count = 0;
    int other_state_count = 0;
    
    for (int i = 0; i < 5; i++) {
        try {
            if (i == 0) throw StateCorruptError();
            if (i == 1) throw NotFoundError();
            if (i == 2) throw AlreadyInitializedError();
            if (i == 3) throw EmptyError();
            if (i == 4) throw ConcurrentModificationError();
        } catch (const StateCorruptError& e) {
            corrupt_count++;
        } catch (const NotFoundError& e) {
            not_found_count++;
        } catch (const StateError& e) {
            other_state_count++;
        }
    }
    
    EXPECT_EQ(1, corrupt_count);
    EXPECT_EQ(1, not_found_count);
    EXPECT_EQ(3, other_state_count);
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorHierarchyTest, AllStateErrorsCatchableAsBase) {
    bool all_caught = true;
    
    try {
        throw StateCorruptError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw AlreadyInitializedError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw NotFoundError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw EmptyError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw ConcurrentModificationError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    EXPECT_TRUE(all_caught);
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorHierarchyTest, DifferentiateStateFromOtherErrors) {
    bool caught_correctly = true;
    
    // Should catch as StateError
    try {
        throw NotFoundError();
    } catch (const StateError&) {
        // Success
    } catch (const ArgumentError&) {
        caught_correctly = false;
    } catch (const MemoryError&) {
        caught_correctly = false;
    }
    
    // Should NOT catch ArgumentError as StateError
    try {
        throw NullPointerError();
    } catch (const StateError&) {
        caught_correctly = false;
    } catch (const ArgumentError&) {
        // Success
    }
    
    // Should NOT catch MemoryError as StateError
    try {
        throw BadAllocError();
    } catch (const StateError&) {
        caught_correctly = false;
    } catch (const MemoryError&) {
        // Success
    }
    
    EXPECT_TRUE(caught_correctly);
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorHierarchyTest, CatchSpecificBeforeGeneral) {
    std::string caught_type;
    
    try {
        throw EmptyError("specific error");
    } catch (const EmptyError& e) {
        caught_type = "EmptyError";
    } catch (const StateError& e) {
        caught_type = "StateError";
    } catch (const Error& e) {
        caught_type = "Error";
    }
    
    EXPECT_EQ("EmptyError", caught_type);
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorHierarchyTest, NotFoundInDataStructure) {
    // Simulate a common use case: searching in a data structure
    auto searchArray = [](int* arr, size_t size, int target) {
        for (size_t i = 0; i < size; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        throw NotFoundError("Target element not found in array");
    };
    
    int data[] = {1, 2, 3, 4, 5};
    
    // Should find element
    EXPECT_EQ(2, searchArray(data, 5, 3));
    
    // Should throw NotFoundError
    try {
        searchArray(data, 5, 10);
        FAIL() << "Expected NotFoundError to be thrown";
    } catch (const NotFoundError& e) {
        EXPECT_STREQ("Target element not found in array", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(StateErrorHierarchyTest, EmptyContainerOperation) {
    // Simulate a common use case: operating on empty container
    auto popFromStack = [](int* stack, size_t& size) -> int {
        if (size == 0) {
            throw EmptyError("Cannot pop from empty stack");
        }
        return stack[--size];
    };
    
    int stack[10] = {1, 2, 3};
    size_t size = 3;
    
    // Should pop successfully
    EXPECT_EQ(3, popFromStack(stack, size));
    EXPECT_EQ(2, popFromStack(stack, size));
    EXPECT_EQ(1, popFromStack(stack, size));
    
    // Should throw EmptyError
    try {
        popFromStack(stack, size);
        FAIL() << "Expected EmptyError to be thrown";
    } catch (const EmptyError& e) {
        EXPECT_STREQ("Cannot pop from empty stack", e.what());
    }
}
// ================================================================================ 
// ================================================================================ 

TEST(DivByZeroErrorTest, DefaultConstructor) {
    DivByZeroError err;
    EXPECT_STREQ("Division by zero", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(DivByZeroErrorTest, CustomMessage) {
    DivByZeroError err("Cannot divide 10 by 0");
    EXPECT_STREQ("Cannot divide 10 by 0", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(DivByZeroErrorTest, ThrowAndCatchSpecific) {
    try {
        throw DivByZeroError("Attempted division by zero in calculation");
    } catch (const DivByZeroError& e) {
        EXPECT_STREQ("Attempted division by zero in calculation", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(DivByZeroErrorTest, CatchAsMathError) {
    try {
        throw DivByZeroError();
    } catch (const MathError& e) {
        EXPECT_STREQ("Division by zero", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(DivByZeroErrorTest, CatchAsBaseError) {
    try {
        throw DivByZeroError();
    } catch (const Error& e) {
        EXPECT_STREQ("Division by zero", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(SingularMatrixErrorTest, DefaultConstructor) {
    SingularMatrixError err;
    EXPECT_STREQ("Singular/non-invertible matrix", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(SingularMatrixErrorTest, CustomMessage) {
    SingularMatrixError err("Matrix determinant is zero, cannot invert");
    EXPECT_STREQ("Matrix determinant is zero, cannot invert", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(SingularMatrixErrorTest, ThrowAndCatchSpecific) {
    try {
        throw SingularMatrixError("Matrix is singular");
    } catch (const SingularMatrixError& e) {
        EXPECT_STREQ("Matrix is singular", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(SingularMatrixErrorTest, CatchAsMathError) {
    try {
        throw SingularMatrixError();
    } catch (const MathError& e) {
        EXPECT_STREQ("Singular/non-invertible matrix", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(SingularMatrixErrorTest, CatchAsBaseError) {
    try {
        throw SingularMatrixError();
    } catch (const Error& e) {
        EXPECT_STREQ("Singular/non-invertible matrix", e.what());
    }
}
// ================================================================================ 
// ================================================================================

TEST(NumericOverflowErrorTest, DefaultConstructor) {
    NumericOverflowError err;
    EXPECT_STREQ("Numeric overflow/underflow", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NumericOverflowErrorTest, CustomMessage) {
    NumericOverflowError err("Result exceeds maximum double value");
    EXPECT_STREQ("Result exceeds maximum double value", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(NumericOverflowErrorTest, ThrowAndCatchSpecific) {
    try {
        throw NumericOverflowError("Integer overflow detected");
    } catch (const NumericOverflowError& e) {
        EXPECT_STREQ("Integer overflow detected", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NumericOverflowErrorTest, CatchAsMathError) {
    try {
        throw NumericOverflowError();
    } catch (const MathError& e) {
        EXPECT_STREQ("Numeric overflow/underflow", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(NumericOverflowErrorTest, CatchAsBaseError) {
    try {
        throw NumericOverflowError();
    } catch (const Error& e) {
        EXPECT_STREQ("Numeric overflow/underflow", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(DomainErrorTest, DefaultConstructor) {
    DomainError err;
    EXPECT_STREQ("Math domain error", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(DomainErrorTest, CustomMessage) {
    DomainError err("Cannot compute sqrt of negative number: -4");
    EXPECT_STREQ("Cannot compute sqrt of negative number: -4", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(DomainErrorTest, ThrowAndCatchSpecific) {
    try {
        throw DomainError("Logarithm of negative number");
    } catch (const DomainError& e) {
        EXPECT_STREQ("Logarithm of negative number", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(DomainErrorTest, CatchAsMathError) {
    try {
        throw DomainError();
    } catch (const MathError& e) {
        EXPECT_STREQ("Math domain error", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(DomainErrorTest, CatchAsBaseError) {
    try {
        throw DomainError();
    } catch (const Error& e) {
        EXPECT_STREQ("Math domain error", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(LossOfPrecisionErrorTest, DefaultConstructor) {
    LossOfPrecisionError err;
    EXPECT_STREQ("Loss of numeric precision", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(LossOfPrecisionErrorTest, CustomMessage) {
    LossOfPrecisionError err("Ill-conditioned matrix: condition number > 1e15");
    EXPECT_STREQ("Ill-conditioned matrix: condition number > 1e15", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(LossOfPrecisionErrorTest, ThrowAndCatchSpecific) {
    try {
        throw LossOfPrecisionError("Catastrophic cancellation detected");
    } catch (const LossOfPrecisionError& e) {
        EXPECT_STREQ("Catastrophic cancellation detected", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(LossOfPrecisionErrorTest, CatchAsMathError) {
    try {
        throw LossOfPrecisionError();
    } catch (const MathError& e) {
        EXPECT_STREQ("Loss of numeric precision", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(LossOfPrecisionErrorTest, CatchAsBaseError) {
    try {
        throw LossOfPrecisionError();
    } catch (const Error& e) {
        EXPECT_STREQ("Loss of numeric precision", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(MathErrorHierarchyTest, CatchMultipleMathErrorTypes) {
    int div_zero_count = 0;
    int domain_count = 0;
    int other_math_count = 0;
    
    for (int i = 0; i < 5; i++) {
        try {
            if (i == 0) throw DivByZeroError();
            if (i == 1) throw DomainError();
            if (i == 2) throw SingularMatrixError();
            if (i == 3) throw NumericOverflowError();
            if (i == 4) throw LossOfPrecisionError();
        } catch (const DivByZeroError& e) {
            div_zero_count++;
        } catch (const DomainError& e) {
            domain_count++;
        } catch (const MathError& e) {
            other_math_count++;
        }
    }
    
    EXPECT_EQ(1, div_zero_count);
    EXPECT_EQ(1, domain_count);
    EXPECT_EQ(3, other_math_count);
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, AllMathErrorsCatchableAsBase) {
    bool all_caught = true;
    
    try {
        throw DivByZeroError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw SingularMatrixError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw NumericOverflowError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw DomainError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw LossOfPrecisionError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    EXPECT_TRUE(all_caught);
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, DifferentiateMathFromOtherErrors) {
    bool caught_correctly = true;
    
    // Should catch as MathError
    try {
        throw DivByZeroError();
    } catch (const MathError&) {
        // Success
    } catch (const ArgumentError&) {
        caught_correctly = false;
    } catch (const MemoryError&) {
        caught_correctly = false;
    } catch (const StateError&) {
        caught_correctly = false;
    }
    
    // Should NOT catch ArgumentError as MathError
    try {
        throw NullPointerError();
    } catch (const MathError&) {
        caught_correctly = false;
    } catch (const ArgumentError&) {
        // Success
    }
    
    // Should NOT catch StateError as MathError
    try {
        throw NotFoundError();
    } catch (const MathError&) {
        caught_correctly = false;
    } catch (const StateError&) {
        // Success
    }
    
    EXPECT_TRUE(caught_correctly);
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, CatchSpecificBeforeGeneral) {
    std::string caught_type;
    
    try {
        throw DivByZeroError("specific error");
    } catch (const DivByZeroError& e) {
        caught_type = "DivByZeroError";
    } catch (const MathError& e) {
        caught_type = "MathError";
    } catch (const Error& e) {
        caught_type = "Error";
    }
    
    EXPECT_EQ("DivByZeroError", caught_type);
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, DivisionOperation) {
    // Simulate a common use case: safe division
    auto safeDivide = [](double numerator, double denominator) -> double {
        if (denominator == 0.0) {
            throw DivByZeroError("Cannot divide by zero");
        }
        return numerator / denominator;
    };
    
    // Should compute successfully
    EXPECT_DOUBLE_EQ(5.0, safeDivide(10.0, 2.0));
    EXPECT_DOUBLE_EQ(-2.5, safeDivide(5.0, -2.0));
    
    // Should throw DivByZeroError
    try {
        safeDivide(10.0, 0.0);
        FAIL() << "Expected DivByZeroError to be thrown";
    } catch (const DivByZeroError& e) {
        EXPECT_STREQ("Cannot divide by zero", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, DomainValidation) {
    // Simulate a common use case: domain validation for sqrt
    auto safeSqrt = [](double x) -> double {
        if (x < 0.0) {
            throw DomainError("Cannot compute square root of negative number");
        }
        // Simplified - real implementation would use std::sqrt
        return x; // Placeholder
    };
    
    // Should compute successfully
    EXPECT_DOUBLE_EQ(4.0, safeSqrt(4.0));
    EXPECT_DOUBLE_EQ(0.0, safeSqrt(0.0));
    
    // Should throw DomainError
    try {
        safeSqrt(-4.0);
        FAIL() << "Expected DomainError to be thrown";
    } catch (const DomainError& e) {
        EXPECT_STREQ("Cannot compute square root of negative number", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, MatrixInversion) {
    // Simulate a common use case: matrix operations
    auto checkMatrixInvertible = [](double determinant) {
        if (determinant == 0.0) {
            throw SingularMatrixError("Matrix is singular: determinant = 0");
        }
        // Proceed with inversion...
    };
    
    // Should succeed
    EXPECT_NO_THROW(checkMatrixInvertible(5.0));
    EXPECT_NO_THROW(checkMatrixInvertible(-2.5));
    
    // Should throw SingularMatrixError
    try {
        checkMatrixInvertible(0.0);
        FAIL() << "Expected SingularMatrixError to be thrown";
    } catch (const SingularMatrixError& e) {
        EXPECT_STREQ("Matrix is singular: determinant = 0", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(MathErrorHierarchyTest, OverflowDetection) {
    // Simulate overflow checking
    auto checkedMultiply = [](int a, int b) -> int {
        // Simplified overflow check
        if (a > 0 && b > 0 && a > (INT_MAX / b)) {
            throw NumericOverflowError("Integer multiplication would overflow");
        }
        return a * b;
    };
    
    // Should compute successfully
    EXPECT_EQ(20, checkedMultiply(4, 5));
    
    // Should throw NumericOverflowError for large values
    try {
        checkedMultiply(1000000, 1000000);
        FAIL() << "Expected NumericOverflowError to be thrown";
    } catch (const NumericOverflowError& e) {
        EXPECT_STREQ("Integer multiplication would overflow", e.what());
    }
}
// ================================================================================ 
// ================================================================================ 

TEST(FileOpenErrorTest, DefaultConstructor) {
    FileOpenError err;
    EXPECT_STREQ("Failed to open file/handle", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileOpenErrorTest, CustomMessage) {
    FileOpenError err("Cannot open config.txt: file does not exist");
    EXPECT_STREQ("Cannot open config.txt: file does not exist", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileOpenErrorTest, ThrowAndCatchSpecific) {
    try {
        throw FileOpenError("File not found");
    } catch (const FileOpenError& e) {
        EXPECT_STREQ("File not found", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileOpenErrorTest, CatchAsIOError) {
    try {
        throw FileOpenError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Failed to open file/handle", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileOpenErrorTest, CatchAsBaseError) {
    try {
        throw FileOpenError();
    } catch (const Error& e) {
        EXPECT_STREQ("Failed to open file/handle", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(FileReadErrorTest, DefaultConstructor) {
    FileReadError err;
    EXPECT_STREQ("Error reading from file/handle", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileReadErrorTest, CustomMessage) {
    FileReadError err("Read operation failed after 512 bytes");
    EXPECT_STREQ("Read operation failed after 512 bytes", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileReadErrorTest, ThrowAndCatchSpecific) {
    try {
        throw FileReadError("Unexpected end of file");
    } catch (const FileReadError& e) {
        EXPECT_STREQ("Unexpected end of file", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileReadErrorTest, CatchAsIOError) {
    try {
        throw FileReadError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Error reading from file/handle", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileReadErrorTest, CatchAsBaseError) {
    try {
        throw FileReadError();
    } catch (const Error& e) {
        EXPECT_STREQ("Error reading from file/handle", e.what());
    }
}
// ================================================================================
// ================================================================================
TEST(FileWriteErrorTest, DefaultConstructor) {
    FileWriteError err;
    EXPECT_STREQ("Error writing to file/handle", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileWriteErrorTest, CustomMessage) {
    FileWriteError err("Disk full: cannot write data");
    EXPECT_STREQ("Disk full: cannot write data", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(FileWriteErrorTest, ThrowAndCatchSpecific) {
    try {
        throw FileWriteError("Write failed");
    } catch (const FileWriteError& e) {
        EXPECT_STREQ("Write failed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileWriteErrorTest, CatchAsIOError) {
    try {
        throw FileWriteError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Error writing to file/handle", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(FileWriteErrorTest, CatchAsBaseError) {
    try {
        throw FileWriteError();
    } catch (const Error& e) {
        EXPECT_STREQ("Error writing to file/handle", e.what());
    }
}
// ================================================================================
// ================================================================================
TEST(PermissionDeniedErrorTest, DefaultConstructor) {
    PermissionDeniedError err;
    EXPECT_STREQ("Permission denied", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PermissionDeniedErrorTest, CustomMessage) {
    PermissionDeniedError err("No write access to /etc/config");
    EXPECT_STREQ("No write access to /etc/config", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(PermissionDeniedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw PermissionDeniedError("Access denied");
    } catch (const PermissionDeniedError& e) {
        EXPECT_STREQ("Access denied", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PermissionDeniedErrorTest, CatchAsIOError) {
    try {
        throw PermissionDeniedError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Permission denied", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(PermissionDeniedErrorTest, CatchAsBaseError) {
    try {
        throw PermissionDeniedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Permission denied", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(IOInterruptedErrorTest, DefaultConstructor) {
    IOInterruptedError err;
    EXPECT_STREQ("I/O interrupted", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOInterruptedErrorTest, CustomMessage) {
    IOInterruptedError err("Read interrupted by signal");
    EXPECT_STREQ("Read interrupted by signal", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOInterruptedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IOInterruptedError("EINTR received");
    } catch (const IOInterruptedError& e) {
        EXPECT_STREQ("EINTR received", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOInterruptedErrorTest, CatchAsIOError) {
    try {
        throw IOInterruptedError();
    } catch (const IOError& e) {
        EXPECT_STREQ("I/O interrupted", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOInterruptedErrorTest, CatchAsBaseError) {
    try {
        throw IOInterruptedError();
    } catch (const Error& e) {
        EXPECT_STREQ("I/O interrupted", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(IOTimeoutErrorTest, DefaultConstructor) {
    IOTimeoutError err;
    EXPECT_STREQ("I/O timed out", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOTimeoutErrorTest, CustomMessage) {
    IOTimeoutError err("Network read timed out after 30 seconds");
    EXPECT_STREQ("Network read timed out after 30 seconds", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOTimeoutErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IOTimeoutError("Operation timeout");
    } catch (const IOTimeoutError& e) {
        EXPECT_STREQ("Operation timeout", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOTimeoutErrorTest, CatchAsIOError) {
    try {
        throw IOTimeoutError();
    } catch (const IOError& e) {
        EXPECT_STREQ("I/O timed out", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOTimeoutErrorTest, CatchAsBaseError) {
    try {
        throw IOTimeoutError();
    } catch (const Error& e) {
        EXPECT_STREQ("I/O timed out", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(IOClosedErrorTest, DefaultConstructor) {
    IOClosedError err;
    EXPECT_STREQ("Operation on closed stream/descriptor", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOClosedErrorTest, CustomMessage) {
    IOClosedError err("Cannot read from closed socket");
    EXPECT_STREQ("Cannot read from closed socket", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOClosedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IOClosedError("Stream already closed");
    } catch (const IOClosedError& e) {
        EXPECT_STREQ("Stream already closed", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOClosedErrorTest, CatchAsIOError) {
    try {
        throw IOClosedError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Operation on closed stream/descriptor", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOClosedErrorTest, CatchAsBaseError) {
    try {
        throw IOClosedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Operation on closed stream/descriptor", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(IOWouldBlockErrorTest, DefaultConstructor) {
    IOWouldBlockError err;
    EXPECT_STREQ("Operation would block", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOWouldBlockErrorTest, CustomMessage) {
    IOWouldBlockError err("Socket read would block in non-blocking mode");
    EXPECT_STREQ("Socket read would block in non-blocking mode", err.what());
}
// -------------------------------------------------------------------------------- 

TEST(IOWouldBlockErrorTest, ThrowAndCatchSpecific) {
    try {
        throw IOWouldBlockError("EWOULDBLOCK received");
    } catch (const IOWouldBlockError& e) {
        EXPECT_STREQ("EWOULDBLOCK received", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOWouldBlockErrorTest, CatchAsIOError) {
    try {
        throw IOWouldBlockError();
    } catch (const IOError& e) {
        EXPECT_STREQ("Operation would block", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOWouldBlockErrorTest, CatchAsBaseError) {
    try {
        throw IOWouldBlockError();
    } catch (const Error& e) {
        EXPECT_STREQ("Operation would block", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(IOErrorHierarchyTest, CatchMultipleIOErrorTypes) {
    int file_open_count = 0;
    int permission_count = 0;
    int other_io_count = 0;
    
    for (int i = 0; i < 6; i++) {
        try {
            if (i == 0) throw FileOpenError();
            if (i == 1) throw PermissionDeniedError();
            if (i == 2) throw FileReadError();
            if (i == 3) throw FileWriteError();
            if (i == 4) throw IOTimeoutError();
            if (i == 5) throw IOClosedError();
        } catch (const FileOpenError& e) {
            file_open_count++;
        } catch (const PermissionDeniedError& e) {
            permission_count++;
        } catch (const IOError& e) {
            other_io_count++;
        }
    }
    
    EXPECT_EQ(1, file_open_count);
    EXPECT_EQ(1, permission_count);
    EXPECT_EQ(4, other_io_count);
}

TEST(IOErrorHierarchyTest, AllIOErrorsCatchableAsBase) {
    bool all_caught = true;
    
    try {
        throw FileOpenError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw FileReadError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw FileWriteError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw PermissionDeniedError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IOInterruptedError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IOTimeoutError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IOClosedError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    try {
        throw IOWouldBlockError();
    } catch (const Error&) {
        // Success
    } catch (...) {
        all_caught = false;
    }
    
    EXPECT_TRUE(all_caught);
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorHierarchyTest, DifferentiateIOFromOtherErrors) {
    bool caught_correctly = true;
    
    // Should catch as IOError
    try {
        throw FileOpenError();
    } catch (const IOError&) {
        // Success
    } catch (const ArgumentError&) {
        caught_correctly = false;
    } catch (const MemoryError&) {
        caught_correctly = false;
    } catch (const StateError&) {
        caught_correctly = false;
    } catch (const MathError&) {
        caught_correctly = false;
    }
    
    // Should NOT catch MathError as IOError
    try {
        throw DivByZeroError();
    } catch (const IOError&) {
        caught_correctly = false;
    } catch (const MathError&) {
        // Success
    }
    
    EXPECT_TRUE(caught_correctly);
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorHierarchyTest, CatchSpecificBeforeGeneral) {
    std::string caught_type;
    
    try {
        throw FileReadError("specific error");
    } catch (const FileReadError& e) {
        caught_type = "FileReadError";
    } catch (const IOError& e) {
        caught_type = "IOError";
    } catch (const Error& e) {
        caught_type = "Error";
    }
    
    EXPECT_EQ("FileReadError", caught_type);
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorHierarchyTest, FileOperationSimulation) {
    // Simulate file opening
    auto openFile = [](const char* filename, bool exists, bool hasPermission) {
        (void)filename;
        if (!exists) {
            throw FileOpenError("File does not exist");
        }
        if (!hasPermission) {
            throw PermissionDeniedError("No read permission");
        }
        return true; // Success
    };
    
    // Should succeed
    EXPECT_TRUE(openFile("test.txt", true, true));
    
    // Should throw FileOpenError
    try {
        openFile("missing.txt", false, true);
        FAIL() << "Expected FileOpenError to be thrown";
    } catch (const FileOpenError& e) {
        EXPECT_STREQ("File does not exist", e.what());
    }
    
    // Should throw PermissionDeniedError
    try {
        openFile("protected.txt", true, false);
        FAIL() << "Expected PermissionDeniedError to be thrown";
    } catch (const PermissionDeniedError& e) {
        EXPECT_STREQ("No read permission", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorHierarchyTest, ReadWriteOperations) {
    // Simulate read/write operations
    auto performIO = [](bool isRead, bool isOpen, bool wouldBlock) {
        if (!isOpen) {
            throw IOClosedError("Stream is closed");
        }
        if (wouldBlock) {
            throw IOWouldBlockError("Non-blocking operation would block");
        }
        if (isRead) {
            // Simulate read failure
            throw FileReadError("Read error occurred");
        } else {
            // Simulate write failure
            throw FileWriteError("Write error occurred");
        }
    };
    
    // Should throw IOClosedError
    try {
        performIO(true, false, false);
        FAIL() << "Expected IOClosedError to be thrown";
    } catch (const IOClosedError& e) {
        EXPECT_STREQ("Stream is closed", e.what());
    }
    
    // Should throw IOWouldBlockError
    try {
        performIO(true, true, true);
        FAIL() << "Expected IOWouldBlockError to be thrown";
    } catch (const IOWouldBlockError& e) {
        EXPECT_STREQ("Non-blocking operation would block", e.what());
    }
    
    // Should throw FileReadError
    try {
        performIO(true, true, false);
        FAIL() << "Expected FileReadError to be thrown";
    } catch (const FileReadError& e) {
        EXPECT_STREQ("Read error occurred", e.what());
    }
    
    // Should throw FileWriteError
    try {
        performIO(false, true, false);
        FAIL() << "Expected FileWriteError to be thrown";
    } catch (const FileWriteError& e) {
        EXPECT_STREQ("Write error occurred", e.what());
    }
}
// -------------------------------------------------------------------------------- 

TEST(IOErrorHierarchyTest, TimeoutScenario) {
    // Simulate timeout handling
    auto waitForData = [](int timeoutMs, int elapsedMs) {
        if (elapsedMs >= timeoutMs) {
            throw IOTimeoutError("Operation exceeded timeout");
        }
        return true;
    };
    
    // Should succeed
    EXPECT_TRUE(waitForData(1000, 500));
    
    // Should throw IOTimeoutError
    try {
        waitForData(1000, 1500);
        FAIL() << "Expected IOTimeoutError to be thrown";
    } catch (const IOTimeoutError& e) {
        EXPECT_STREQ("Operation exceeded timeout", e.what());
    }
}
// ================================================================================ 
// ================================================================================ 

TEST(TypeMismatchErrorTest, DefaultConstructor) {
    TypeMismatchError err;
    EXPECT_STREQ("Type mismatch", err.what());
}
// --------------------------------------------------------------------------------

TEST(TypeMismatchErrorTest, CustomMessage) {
    TypeMismatchError err("Expected integer but received string");
    EXPECT_STREQ("Expected integer but received string", err.what());
}
// --------------------------------------------------------------------------------

TEST(TypeMismatchErrorTest, ThrowAndCatchSpecific) {
    try {
        throw TypeMismatchError("Type mismatch during decoding");
    } catch (const TypeMismatchError& e) {
        EXPECT_STREQ("Type mismatch during decoding", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(TypeMismatchErrorTest, CatchAsFormatError) {
    try {
        throw TypeMismatchError();
    } catch (const FormatError& e) {
        EXPECT_STREQ("Type mismatch", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(TypeMismatchErrorTest, CatchAsBaseError) {
    try {
        throw TypeMismatchError();
    } catch (const Error& e) {
        EXPECT_STREQ("Type mismatch", e.what());
    }
}
// ================================================================================

TEST(FormatInvalidErrorTest, DefaultConstructor) {
    FormatInvalidError err;
    EXPECT_STREQ("Invalid data format", err.what());
}
// --------------------------------------------------------------------------------

TEST(FormatInvalidErrorTest, CustomMessage) {
    FormatInvalidError err("Malformed header detected");
    EXPECT_STREQ("Malformed header detected", err.what());
}
// --------------------------------------------------------------------------------

TEST(FormatInvalidErrorTest, ThrowAndCatchSpecific) {
    try {
        throw FormatInvalidError("Invalid data format in record 12");
    } catch (const FormatInvalidError& e) {
        EXPECT_STREQ("Invalid data format in record 12", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(FormatInvalidErrorTest, CatchAsFormatError) {
    try {
        throw FormatInvalidError();
    } catch (const FormatError& e) {
        EXPECT_STREQ("Invalid data format", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(FormatInvalidErrorTest, CatchAsBaseError) {
    try {
        throw FormatInvalidError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid data format", e.what());
    }
}
// ================================================================================

TEST(EncodingInvalidErrorTest, DefaultConstructor) {
    EncodingInvalidError err;
    EXPECT_STREQ("Invalid text encoding", err.what());
}
// --------------------------------------------------------------------------------

TEST(EncodingInvalidErrorTest, CustomMessage) {
    EncodingInvalidError err("UTF-8 decoding failed");
    EXPECT_STREQ("UTF-8 decoding failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(EncodingInvalidErrorTest, ThrowAndCatchSpecific) {
    try {
        throw EncodingInvalidError("Invalid text encoding encountered");
    } catch (const EncodingInvalidError& e) {
        EXPECT_STREQ("Invalid text encoding encountered", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(EncodingInvalidErrorTest, CatchAsFormatError) {
    try {
        throw EncodingInvalidError();
    } catch (const FormatError& e) {
        EXPECT_STREQ("Invalid text encoding", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(EncodingInvalidErrorTest, CatchAsBaseError) {
    try {
        throw EncodingInvalidError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid text encoding", e.what());
    }
}
// ================================================================================

TEST(ParsingFailedErrorTest, DefaultConstructor) {
    ParsingFailedError err;
    EXPECT_STREQ("Parsing failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(ParsingFailedErrorTest, CustomMessage) {
    ParsingFailedError err("JSON parsing failed at line 12");
    EXPECT_STREQ("JSON parsing failed at line 12", err.what());
}
// --------------------------------------------------------------------------------

TEST(ParsingFailedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ParsingFailedError("Parsing failed while reading config");
    } catch (const ParsingFailedError& e) {
        EXPECT_STREQ("Parsing failed while reading config", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ParsingFailedErrorTest, CatchAsFormatError) {
    try {
        throw ParsingFailedError();
    } catch (const FormatError& e) {
        EXPECT_STREQ("Parsing failed", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ParsingFailedErrorTest, CatchAsBaseError) {
    try {
        throw ParsingFailedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Parsing failed", e.what());
    }
}
// ================================================================================

TEST(ValidationFailedErrorTest, DefaultConstructor) {
    ValidationFailedError err;
    EXPECT_STREQ("Validation failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(ValidationFailedErrorTest, CustomMessage) {
    ValidationFailedError err("Checksum validation failed");
    EXPECT_STREQ("Checksum validation failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(ValidationFailedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ValidationFailedError("Validation failed for input payload");
    } catch (const ValidationFailedError& e) {
        EXPECT_STREQ("Validation failed for input payload", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ValidationFailedErrorTest, CatchAsFormatError) {
    try {
        throw ValidationFailedError();
    } catch (const FormatError& e) {
        EXPECT_STREQ("Validation failed", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ValidationFailedErrorTest, CatchAsBaseError) {
    try {
        throw ValidationFailedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Validation failed", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(LockFailedErrorTest, DefaultConstructor) {
    LockFailedError err;
    EXPECT_STREQ("Lock operation failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(LockFailedErrorTest, CustomMessage) {
    LockFailedError err("Failed to acquire mutex");
    EXPECT_STREQ("Failed to acquire mutex", err.what());
}
// --------------------------------------------------------------------------------

TEST(LockFailedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw LockFailedError("Lock operation failed while entering critical section");
    } catch (const LockFailedError& e) {
        EXPECT_STREQ("Lock operation failed while entering critical section", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(LockFailedErrorTest, CatchAsConcurrencyError) {
    try {
        throw LockFailedError();
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Lock operation failed", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(LockFailedErrorTest, CatchAsBaseError) {
    try {
        throw LockFailedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Lock operation failed", e.what());
    }
}
// ================================================================================

TEST(DeadlockDetectedErrorTest, DefaultConstructor) {
    DeadlockDetectedError err;
    EXPECT_STREQ("Deadlock detected", err.what());
}
// --------------------------------------------------------------------------------

TEST(DeadlockDetectedErrorTest, CustomMessage) {
    DeadlockDetectedError err("Deadlock detected between worker threads");
    EXPECT_STREQ("Deadlock detected between worker threads", err.what());
}
// --------------------------------------------------------------------------------

TEST(DeadlockDetectedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw DeadlockDetectedError("Deadlock detected while locking resources");
    } catch (const DeadlockDetectedError& e) {
        EXPECT_STREQ("Deadlock detected while locking resources", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(DeadlockDetectedErrorTest, CatchAsConcurrencyError) {
    try {
        throw DeadlockDetectedError();
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Deadlock detected", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(DeadlockDetectedErrorTest, CatchAsBaseError) {
    try {
        throw DeadlockDetectedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Deadlock detected", e.what());
    }
}
// ================================================================================

TEST(ThreadFailErrorTest, DefaultConstructor) {
    ThreadFailError err;
    EXPECT_STREQ("Thread operation failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(ThreadFailErrorTest, CustomMessage) {
    ThreadFailError err("Thread creation failed");
    EXPECT_STREQ("Thread creation failed", err.what());
}
// --------------------------------------------------------------------------------

TEST(ThreadFailErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ThreadFailError("Thread operation failed during join");
    } catch (const ThreadFailError& e) {
        EXPECT_STREQ("Thread operation failed during join", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ThreadFailErrorTest, CatchAsConcurrencyError) {
    try {
        throw ThreadFailError();
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Thread operation failed", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ThreadFailErrorTest, CatchAsBaseError) {
    try {
        throw ThreadFailError();
    } catch (const Error& e) {
        EXPECT_STREQ("Thread operation failed", e.what());
    }
}
// ================================================================================

TEST(CancelledErrorTest, DefaultConstructor) {
    CancelledError err;
    EXPECT_STREQ("Operation cancelled", err.what());
}
// --------------------------------------------------------------------------------

TEST(CancelledErrorTest, CustomMessage) {
    CancelledError err("Operation cancelled by user request");
    EXPECT_STREQ("Operation cancelled by user request", err.what());
}
// --------------------------------------------------------------------------------

TEST(CancelledErrorTest, ThrowAndCatchSpecific) {
    try {
        throw CancelledError("Operation cancelled before completion");
    } catch (const CancelledError& e) {
        EXPECT_STREQ("Operation cancelled before completion", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(CancelledErrorTest, CatchAsConcurrencyError) {
    try {
        throw CancelledError();
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Operation cancelled", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(CancelledErrorTest, CatchAsBaseError) {
    try {
        throw CancelledError();
    } catch (const Error& e) {
        EXPECT_STREQ("Operation cancelled", e.what());
    }
}
// ================================================================================

TEST(RaceDetectedErrorTest, DefaultConstructor) {
    RaceDetectedError err;
    EXPECT_STREQ("Data race detected", err.what());
}
// --------------------------------------------------------------------------------

TEST(RaceDetectedErrorTest, CustomMessage) {
    RaceDetectedError err("Concurrent write detected on shared buffer");
    EXPECT_STREQ("Concurrent write detected on shared buffer", err.what());
}
// --------------------------------------------------------------------------------

TEST(RaceDetectedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw RaceDetectedError("Data race detected during parallel update");
    } catch (const RaceDetectedError& e) {
        EXPECT_STREQ("Data race detected during parallel update", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(RaceDetectedErrorTest, CatchAsConcurrencyError) {
    try {
        throw RaceDetectedError();
    } catch (const ConcurrencyError& e) {
        EXPECT_STREQ("Data race detected", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(RaceDetectedErrorTest, CatchAsBaseError) {
    try {
        throw RaceDetectedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Data race detected", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(ConfigInvalidErrorTest, DefaultConstructor) {
    ConfigInvalidError err;
    EXPECT_STREQ("Invalid configuration", err.what());
}
// --------------------------------------------------------------------------------

TEST(ConfigInvalidErrorTest, CustomMessage) {
    ConfigInvalidError err("Missing required configuration key");
    EXPECT_STREQ("Missing required configuration key", err.what());
}
// --------------------------------------------------------------------------------

TEST(ConfigInvalidErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ConfigInvalidError("Invalid configuration for runtime mode");
    } catch (const ConfigInvalidError& e) {
        EXPECT_STREQ("Invalid configuration for runtime mode", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ConfigInvalidErrorTest, CatchAsConfigError) {
    try {
        throw ConfigInvalidError();
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Invalid configuration", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ConfigInvalidErrorTest, CatchAsBaseError) {
    try {
        throw ConfigInvalidError();
    } catch (const Error& e) {
        EXPECT_STREQ("Invalid configuration", e.what());
    }
}
// ================================================================================

TEST(UnsupportedErrorTest, DefaultConstructor) {
    UnsupportedError err;
    EXPECT_STREQ("Unsupported feature/platform", err.what());
}
// --------------------------------------------------------------------------------

TEST(UnsupportedErrorTest, CustomMessage) {
    UnsupportedError err("ARM platform not supported");
    EXPECT_STREQ("ARM platform not supported", err.what());
}
// --------------------------------------------------------------------------------

TEST(UnsupportedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw UnsupportedError("Unsupported feature/platform in current build");
    } catch (const UnsupportedError& e) {
        EXPECT_STREQ("Unsupported feature/platform in current build", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(UnsupportedErrorTest, CatchAsConfigError) {
    try {
        throw UnsupportedError();
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Unsupported feature/platform", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(UnsupportedErrorTest, CatchAsBaseError) {
    try {
        throw UnsupportedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Unsupported feature/platform", e.what());
    }
}
// ================================================================================

TEST(FeatureDisabledErrorTest, DefaultConstructor) {
    FeatureDisabledError err;
    EXPECT_STREQ("Feature disabled by policy/build", err.what());
}
// --------------------------------------------------------------------------------

TEST(FeatureDisabledErrorTest, CustomMessage) {
    FeatureDisabledError err("Feature disabled by security policy");
    EXPECT_STREQ("Feature disabled by security policy", err.what());
}
// --------------------------------------------------------------------------------

TEST(FeatureDisabledErrorTest, ThrowAndCatchSpecific) {
    try {
        throw FeatureDisabledError("Feature disabled by policy/build for this target");
    } catch (const FeatureDisabledError& e) {
        EXPECT_STREQ("Feature disabled by policy/build for this target", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(FeatureDisabledErrorTest, CatchAsConfigError) {
    try {
        throw FeatureDisabledError();
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Feature disabled by policy/build", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(FeatureDisabledErrorTest, CatchAsBaseError) {
    try {
        throw FeatureDisabledError();
    } catch (const Error& e) {
        EXPECT_STREQ("Feature disabled by policy/build", e.what());
    }
}
// ================================================================================

TEST(VersionMismatchErrorTest, DefaultConstructor) {
    VersionMismatchError err;
    EXPECT_STREQ("Version/ABI mismatch", err.what());
}
// --------------------------------------------------------------------------------

TEST(VersionMismatchErrorTest, CustomMessage) {
    VersionMismatchError err("Library ABI version mismatch");
    EXPECT_STREQ("Library ABI version mismatch", err.what());
}
// --------------------------------------------------------------------------------

TEST(VersionMismatchErrorTest, ThrowAndCatchSpecific) {
    try {
        throw VersionMismatchError("Version/ABI mismatch detected at startup");
    } catch (const VersionMismatchError& e) {
        EXPECT_STREQ("Version/ABI mismatch detected at startup", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(VersionMismatchErrorTest, CatchAsConfigError) {
    try {
        throw VersionMismatchError();
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Version/ABI mismatch", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(VersionMismatchErrorTest, CatchAsBaseError) {
    try {
        throw VersionMismatchError();
    } catch (const Error& e) {
        EXPECT_STREQ("Version/ABI mismatch", e.what());
    }
}
// ================================================================================

TEST(ResourceExhaustedErrorTest, DefaultConstructor) {
    ResourceExhaustedError err;
    EXPECT_STREQ("Resource exhausted", err.what());
}
// --------------------------------------------------------------------------------

TEST(ResourceExhaustedErrorTest, CustomMessage) {
    ResourceExhaustedError err("Out of file descriptors");
    EXPECT_STREQ("Out of file descriptors", err.what());
}
// --------------------------------------------------------------------------------

TEST(ResourceExhaustedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw ResourceExhaustedError("Resource exhausted while allocating handles");
    } catch (const ResourceExhaustedError& e) {
        EXPECT_STREQ("Resource exhausted while allocating handles", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ResourceExhaustedErrorTest, CatchAsConfigError) {
    try {
        throw ResourceExhaustedError();
    } catch (const ConfigError& e) {
        EXPECT_STREQ("Resource exhausted", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(ResourceExhaustedErrorTest, CatchAsBaseError) {
    try {
        throw ResourceExhaustedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Resource exhausted", e.what());
    }
}
// ================================================================================
// ================================================================================

TEST(NotImplementedErrorTest, DefaultConstructor) {
    NotImplementedError err;
    EXPECT_STREQ("Not implemented", err.what());
}
// --------------------------------------------------------------------------------

TEST(NotImplementedErrorTest, CustomMessage) {
    NotImplementedError err("Serialization not yet implemented");
    EXPECT_STREQ("Serialization not yet implemented", err.what());
}
// --------------------------------------------------------------------------------

TEST(NotImplementedErrorTest, ThrowAndCatchSpecific) {
    try {
        throw NotImplementedError("Not implemented for this backend");
    } catch (const NotImplementedError& e) {
        EXPECT_STREQ("Not implemented for this backend", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(NotImplementedErrorTest, CatchAsGenericError) {
    try {
        throw NotImplementedError();
    } catch (const GenericError& e) {
        EXPECT_STREQ("Not implemented", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(NotImplementedErrorTest, CatchAsBaseError) {
    try {
        throw NotImplementedError();
    } catch (const Error& e) {
        EXPECT_STREQ("Not implemented", e.what());
    }
}
// ================================================================================

TEST(OperationUnavailableErrorTest, DefaultConstructor) {
    OperationUnavailableError err;
    EXPECT_STREQ("Operation unavailable", err.what());
}
// --------------------------------------------------------------------------------

TEST(OperationUnavailableErrorTest, CustomMessage) {
    OperationUnavailableError err("Service unavailable during shutdown");
    EXPECT_STREQ("Service unavailable during shutdown", err.what());
}
// --------------------------------------------------------------------------------

TEST(OperationUnavailableErrorTest, ThrowAndCatchSpecific) {
    try {
        throw OperationUnavailableError("Operation unavailable while initializing");
    } catch (const OperationUnavailableError& e) {
        EXPECT_STREQ("Operation unavailable while initializing", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(OperationUnavailableErrorTest, CatchAsGenericError) {
    try {
        throw OperationUnavailableError();
    } catch (const GenericError& e) {
        EXPECT_STREQ("Operation unavailable", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(OperationUnavailableErrorTest, CatchAsBaseError) {
    try {
        throw OperationUnavailableError();
    } catch (const Error& e) {
        EXPECT_STREQ("Operation unavailable", e.what());
    }
}
// ================================================================================

TEST(UnknownErrorTest, DefaultConstructor) {
    UnknownError err;
    EXPECT_STREQ("Unknown error", err.what());
}
// --------------------------------------------------------------------------------

TEST(UnknownErrorTest, CustomMessage) {
    UnknownError err("Unexpected failure occurred");
    EXPECT_STREQ("Unexpected failure occurred", err.what());
}
// --------------------------------------------------------------------------------

TEST(UnknownErrorTest, ThrowAndCatchSpecific) {
    try {
        throw UnknownError("Unknown error while processing request");
    } catch (const UnknownError& e) {
        EXPECT_STREQ("Unknown error while processing request", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(UnknownErrorTest, CatchAsGenericError) {
    try {
        throw UnknownError();
    } catch (const GenericError& e) {
        EXPECT_STREQ("Unknown error", e.what());
    }
}
// --------------------------------------------------------------------------------

TEST(UnknownErrorTest, CatchAsBaseError) {
    try {
        throw UnknownError();
    } catch (const Error& e) {
        EXPECT_STREQ("Unknown error", e.what());
    }
}
// ================================================================================ 
// ================================================================================ 
// Expected class 

TEST(ExpectedTest, DefaultConstruction) {
    Expected<int> result;
    
    EXPECT_FALSE(result.hasValue());
    EXPECT_TRUE(result.hasError());
}

TEST(ExpectedTest, SetValue) {
    Expected<int> result;
    result.setValue(42);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_FALSE(result.hasError());
    EXPECT_EQ(42, result.value());
}

TEST(ExpectedTest, SetError) {
    Expected<int> result;
    result.setError(Error("Test error"));
    
    EXPECT_FALSE(result.hasValue());
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Test error", result.error().what());
}

TEST(ExpectedTest, SetValueThenError) {
    Expected<int> result;
    result.setValue(42);
    EXPECT_TRUE(result.hasValue());
    
    result.setError(Error("Something went wrong"));
    EXPECT_FALSE(result.hasValue());
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Something went wrong", result.error().what());
}

TEST(ExpectedTest, SetErrorThenValue) {
    Expected<int> result;
    result.setError(Error("Initial error"));
    EXPECT_TRUE(result.hasError());
    
    result.setValue(100);
    EXPECT_TRUE(result.hasValue());
    EXPECT_FALSE(result.hasError());
    EXPECT_EQ(100, result.value());
}

// ================================================================================
// Copy Constructor and Assignment Tests
// ================================================================================
TEST(ExpectedTest, CopyConstructorWithValue) {
    Expected<int> original;
    original.setValue(42);
    
    Expected<int> copy(original);
    
    EXPECT_TRUE(copy.hasValue());
    EXPECT_EQ(42, copy.value());
    
    // Ensure independence
    copy.setValue(100);
    EXPECT_EQ(42, original.value());
    EXPECT_EQ(100, copy.value());
}

TEST(ExpectedTest, CopyConstructorWithError) {
    Expected<int> original;
    original.setError(Error("Test error"));
    
    Expected<int> copy(original);
    
    EXPECT_TRUE(copy.hasError());
    EXPECT_STREQ("Test error", copy.error().what());
}

TEST(ExpectedTest, CopyAssignmentValueToValue) {
    Expected<int> a;
    a.setValue(10);
    
    Expected<int> b;
    b.setValue(20);
    
    a = b;
    
    EXPECT_TRUE(a.hasValue());
    EXPECT_EQ(20, a.value());
}

TEST(ExpectedTest, CopyAssignmentErrorToError) {
    Expected<int> a;
    a.setError(Error("Error A"));
    
    Expected<int> b;
    b.setError(Error("Error B"));
    
    a = b;
    
    EXPECT_TRUE(a.hasError());
    EXPECT_STREQ("Error B", a.error().what());
}

TEST(ExpectedTest, CopyAssignmentValueToError) {
    Expected<int> a;
    a.setValue(42);
    
    Expected<int> b;
    b.setError(Error("Error B"));
    
    a = b;
    
    EXPECT_TRUE(a.hasError());
    EXPECT_STREQ("Error B", a.error().what());
}

TEST(ExpectedTest, CopyAssignmentErrorToValue) {
    Expected<int> a;
    a.setError(Error("Error A"));
    
    Expected<int> b;
    b.setValue(42);
    
    a = b;
    
    EXPECT_TRUE(a.hasValue());
    EXPECT_EQ(42, a.value());
}

// ================================================================================
// Bool Conversion Tests
// ================================================================================
TEST(ExpectedTest, BoolConversionWithValue) {
    Expected<int> result;
    result.setValue(42);
    
    EXPECT_TRUE(static_cast<bool>(result));
    if (result) {
        EXPECT_EQ(42, result.value());
    } else {
        FAIL() << "Expected result to be true";
    }
}

TEST(ExpectedTest, BoolConversionWithError) {
    Expected<int> result;
    result.setError(Error("Test error"));
    
    EXPECT_FALSE(static_cast<bool>(result));
    if (!result) {
        EXPECT_STREQ("Test error", result.error().what());
    } else {
        FAIL() << "Expected result to be false";
    }
}

// ================================================================================
// ValueOr Tests
// ================================================================================
TEST(ExpectedTest, ValueOrWithValue) {
    Expected<int> result;
    result.setValue(42);
    
    EXPECT_EQ(42, result.valueOr(0));
    EXPECT_EQ(42, result.valueOr(-1));
}

TEST(ExpectedTest, ValueOrWithError) {
    Expected<int> result;
    result.setError(Error("Test error"));
    
    EXPECT_EQ(0, result.valueOr(0));
    EXPECT_EQ(-1, result.valueOr(-1));
    EXPECT_EQ(999, result.valueOr(999));
}

// ================================================================================
// Different Value Types Tests
// ================================================================================
TEST(ExpectedTest, DoubleType) {
    Expected<double> result;
    result.setValue(3.14159);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_DOUBLE_EQ(3.14159, result.value());
}

TEST(ExpectedTest, StringType) {
    Expected<const char*> result;
    result.setValue("Hello, World!");
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_STREQ("Hello, World!", result.value());
}

TEST(ExpectedTest, PointerType) {
    int x = 42;
    Expected<int*> result;
    result.setValue(&x);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_EQ(&x, result.value());
    EXPECT_EQ(42, *result.value());
}

TEST(ExpectedTest, BoolType) {
    Expected<bool> result;
    result.setValue(true);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_TRUE(result.value());
}

// ================================================================================
// Error Hierarchy Tests
// ================================================================================
TEST(ExpectedTest, WithArgumentError) {
    Expected<int> result;
    result.setError(NullPointerError("Null pointer encountered"));
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Null pointer encountered", result.error().what());
}

TEST(ExpectedTest, WithMemoryError) {
    Expected<int> result;
    result.setError(BadAllocError("Allocation failed"));
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Allocation failed", result.error().what());
}

TEST(ExpectedTest, WithMathError) {
    Expected<double> result;
    result.setError(DivByZeroError());
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Division by zero", result.error().what());
}

TEST(ExpectedTest, WithIOError) {
    Expected<int> result;
    result.setError(FileOpenError("Cannot open file"));
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Cannot open file", result.error().what());
}

TEST(ExpectedTest, WithStateError) {
    Expected<int> result;
    result.setError(NotFoundError("Item not in collection"));
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Item not in collection", result.error().what());
}

// ================================================================================
// Practical Usage Tests
// ================================================================================
TEST(ExpectedTest, DivisionFunction) {
    auto safeDivide = [](int a, int b) -> Expected<int> {
        Expected<int> result;
        if (b == 0) {
            result.setError(DivByZeroError("Cannot divide by zero"));
            return result;
        }
        result.setValue(a / b);
        return result;
    };
    
    // Successful division
    Expected<int> result1 = safeDivide(10, 2);
    EXPECT_TRUE(result1.hasValue());
    EXPECT_EQ(5, result1.value());
    
    // Division by zero
    Expected<int> result2 = safeDivide(10, 0);
    EXPECT_TRUE(result2.hasError());
    EXPECT_STREQ("Cannot divide by zero", result2.error().what());
    
    // Negative numbers
    Expected<int> result3 = safeDivide(-20, 4);
    EXPECT_TRUE(result3.hasValue());
    EXPECT_EQ(-5, result3.value());
}

TEST(ExpectedTest, FileOperationSimulation) {
    auto openFile = [](const char* filename) -> Expected<int> {
        Expected<int> result;
        if (filename == nullptr) {
            result.setError(NullPointerError("Filename is null"));
            return result;
        }
        if (filename[0] == '\0') {
            result.setError(InvalidArgError("Filename is empty"));
            return result;
        }
        // Simulate file handle
        result.setValue(42);
        return result;
    };
    
    // Success
    Expected<int> result1 = openFile("test.txt");
    EXPECT_TRUE(result1.hasValue());
    EXPECT_EQ(42, result1.value());
    
    // Null pointer
    Expected<int> result2 = openFile(nullptr);
    EXPECT_TRUE(result2.hasError());
    EXPECT_STREQ("Filename is null", result2.error().what());
    
    // Empty filename
    Expected<int> result3 = openFile("");
    EXPECT_TRUE(result3.hasError());
    EXPECT_STREQ("Filename is empty", result3.error().what());
}

TEST(ExpectedTest, ArrayAccessSimulation) {
    auto safeArrayAccess = [](int* arr, size_t size, size_t index) -> Expected<int> {
        Expected<int> result;
        if (arr == nullptr) {
            result.setError(NullPointerError("Array is null"));
            return result;
        }
        if (index >= size) {
            result.setError(OutOfBoundsError("Index out of bounds"));
            return result;
        }
        result.setValue(arr[index]);
        return result;
    };
    
    int data[] = {10, 20, 30, 40, 50};
    
    // Valid access
    Expected<int> result1 = safeArrayAccess(data, 5, 2);
    EXPECT_TRUE(result1.hasValue());
    EXPECT_EQ(30, result1.value());
    
    // Out of bounds
    Expected<int> result2 = safeArrayAccess(data, 5, 10);
    EXPECT_TRUE(result2.hasError());
    EXPECT_STREQ("Index out of bounds", result2.error().what());
    
    // Null array
    Expected<int> result3 = safeArrayAccess(nullptr, 5, 0);
    EXPECT_TRUE(result3.hasError());
    EXPECT_STREQ("Array is null", result3.error().what());
}

TEST(ExpectedTest, ChainedOperations) {
    auto divide = [](int a, int b) -> Expected<int> {
        Expected<int> result;
        if (b == 0) {
            result.setError(DivByZeroError());
            return result;
        }
        result.setValue(a / b);
        return result;
    };
    
    auto multiplyBy2 = [](int x) -> Expected<int> {
        Expected<int> result;
        if (x > 1000000) {
            result.setError(NumericOverflowError("Result too large"));
            return result;
        }
        result.setValue(x * 2);
        return result;
    };
    
    // Success path
    Expected<int> step1 = divide(10, 2);
    if (step1.hasValue()) {
        Expected<int> step2 = multiplyBy2(step1.value());
        EXPECT_TRUE(step2.hasValue());
        EXPECT_EQ(10, step2.value());
    } else {
        FAIL() << "Step 1 should succeed";
    }
    
    // Error in first step
    Expected<int> step3 = divide(10, 0);
    EXPECT_TRUE(step3.hasError());
    
    // Error in second step - need a value > 1000000 after division
    Expected<int> step4 = divide(2000002, 2);  // Results in 1000001
    if (step4.hasValue()) {
        EXPECT_EQ(1000001, step4.value());  // Verify the intermediate value
        Expected<int> step5 = multiplyBy2(step4.value());
        EXPECT_TRUE(step5.hasError());
        EXPECT_STREQ("Result too large", step5.error().what());
    } else {
        FAIL() << "Step 4 should succeed";
    }
}

TEST(ExpectedTest, SquareRootSimulation) {
    auto safeSqrt = [](double x) -> Expected<double> {
        Expected<double> result;
        if (x < 0.0) {
            result.setError(DomainError("Cannot compute square root of negative number"));
            return result;
        }
        // Simplified - real implementation would use std::sqrt
        result.setValue(x);
        return result;
    };
    
    // Valid input
    Expected<double> result1 = safeSqrt(4.0);
    EXPECT_TRUE(result1.hasValue());
    EXPECT_DOUBLE_EQ(4.0, result1.value());
    
    // Negative input
    Expected<double> result2 = safeSqrt(-4.0);
    EXPECT_TRUE(result2.hasError());
    EXPECT_STREQ("Cannot compute square root of negative number", result2.error().what());
    
    // Zero
    Expected<double> result3 = safeSqrt(0.0);
    EXPECT_TRUE(result3.hasValue());
    EXPECT_DOUBLE_EQ(0.0, result3.value());
}

// ================================================================================
// Edge Cases
// ================================================================================
TEST(ExpectedTest, ZeroValue) {
    Expected<int> result;
    result.setValue(0);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_EQ(0, result.value());
    EXPECT_TRUE(static_cast<bool>(result));  // Should still be true even with 0 value
}

TEST(ExpectedTest, NegativeValue) {
    Expected<int> result;
    result.setValue(-42);
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_EQ(-42, result.value());
}

TEST(ExpectedTest, EmptyErrorMessage) {
    Expected<int> result;
    result.setError(Error(""));
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("", result.error().what());
}

TEST(ExpectedTest, LongErrorMessage) {
    char longMsg[300];
    for (int i = 0; i < 299; i++) {
        longMsg[i] = 'A';
    }
    longMsg[299] = '\0';
    
    Expected<int> result;
    result.setError(Error(longMsg));
    
    EXPECT_TRUE(result.hasError());
    // Should be truncated to 255 characters
    EXPECT_EQ(255, strlen(result.error().what()));
}

TEST(ExpectedTest, RepeatedSetValue) {
    Expected<int> result;
    result.setValue(10);
    EXPECT_EQ(10, result.value());
    
    result.setValue(20);
    EXPECT_EQ(20, result.value());
    
    result.setValue(30);
    EXPECT_EQ(30, result.value());
}

TEST(ExpectedTest, RepeatedSetError) {
    Expected<int> result;
    result.setError(Error("Error 1"));
    EXPECT_STREQ("Error 1", result.error().what());
    
    result.setError(Error("Error 2"));
    EXPECT_STREQ("Error 2", result.error().what());
    
    result.setError(Error("Error 3"));
    EXPECT_STREQ("Error 3", result.error().what());
}

// ================================================================================
// Const Correctness Tests
// ================================================================================
TEST(ExpectedTest, ConstWithValue) {
    Expected<int> temp;
    temp.setValue(42);
    const Expected<int> result = temp;
    
    EXPECT_TRUE(result.hasValue());
    EXPECT_EQ(42, result.value());
    EXPECT_TRUE(static_cast<bool>(result));
}

TEST(ExpectedTest, ConstWithError) {
    Expected<int> temp;
    temp.setError(Error("Test error"));
    const Expected<int> result = temp;
    
    EXPECT_TRUE(result.hasError());
    EXPECT_STREQ("Test error", result.error().what());
    EXPECT_FALSE(static_cast<bool>(result));
}

// ================================================================================
// Return Value Optimization Test
// ================================================================================
TEST(ExpectedTest, FunctionReturn) {
    auto createValue = []() -> Expected<int> {
        Expected<int> result;
        result.setValue(100);
        return result;  // Should use copy elision/RVO
    };
    
    auto createError = []() -> Expected<int> {
        Expected<int> result;
        result.setError(Error("Function error"));
        return result;  // Should use copy elision/RVO
    };
    
    Expected<int> val = createValue();
    EXPECT_TRUE(val.hasValue());
    EXPECT_EQ(100, val.value());
    
    Expected<int> err = createError();
    EXPECT_TRUE(err.hasError());
    EXPECT_STREQ("Function error", err.error().what());
}
// ================================================================================
// ================================================================================
// eof
