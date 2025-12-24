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
// eof
