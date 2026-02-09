// ================================================================================
// ================================================================================
// - File:    test_allocator.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    December 28, 2025
// - Version: 1.0
// - Copyright: Copyright 2025, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include "allocator.hpp"
#include "string.hpp"

#include <gtest/gtest.h>
#include <cstring>
// ================================================================================ 
// ================================================================================

// Test Fixture for String Tests
// ================================================================================

class StringTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Setup code if needed
    }
    
    void TearDown() override {
        // Cleanup code if needed
    }
};

// ================================================================================
// Basic Initialization Tests
// ================================================================================

TEST_F(StringTest, InitWithDefaultCapacity) {
    auto result = cslt::String::init("hello", 0, allocator);
    
    ASSERT_TRUE(result.hasValue()) << "String initialization failed";
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("hello", str->c_str());
    EXPECT_EQ(5, str->size());
    EXPECT_EQ(6, str->capacity()); // 5 + null terminator
}

TEST_F(StringTest, InitWithEmptyString) {
    auto result = cslt::String::init("", 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("", str->c_str());
    EXPECT_EQ(0, str->size());
    EXPECT_EQ(1, str->capacity()); // Just null terminator
}

TEST_F(StringTest, InitWithSingleCharacter) {
    auto result = cslt::String::init("x", 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("x", str->c_str());
    EXPECT_EQ(1, str->size());
    EXPECT_EQ(2, str->capacity());
}

TEST_F(StringTest, InitWithLongString) {
    const char* long_str = "This is a much longer string to test memory allocation";
    auto result = cslt::String::init(long_str, 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ(long_str, str->c_str());
    EXPECT_EQ(std::strlen(long_str), str->size());
    EXPECT_EQ(std::strlen(long_str) + 1, str->capacity());
}

// ================================================================================
// Capacity Tests
// ================================================================================

TEST_F(StringTest, InitWithExactCapacity) {
    auto result = cslt::String::init("hello", 5, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("hello", str->c_str());
    EXPECT_EQ(5, str->size());
    EXPECT_EQ(6, str->capacity()); // 5 + null
}

TEST_F(StringTest, InitWithLargerCapacity) {
    auto result = cslt::String::init("hi", 100, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("hi", str->c_str());
    EXPECT_EQ(2, str->size());
    EXPECT_EQ(101, str->capacity()); // 100 + null
}

TEST_F(StringTest, InitWithSmallerCapacityTruncates) {
    auto result = cslt::String::init("hello world", 5, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("hello", str->c_str());
    EXPECT_EQ(5, str->size());
    EXPECT_EQ(6, str->capacity());
}

TEST_F(StringTest, InitWithZeroCapacityAndLongString) {
    const char* text = "abcdefghijklmnopqrstuvwxyz";
    auto result = cslt::String::init(text, 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ(text, str->c_str());
    EXPECT_EQ(26, str->size());
    EXPECT_EQ(27, str->capacity());
}

TEST_F(StringTest, TruncationEnsuresNullTermination) {
    auto result = cslt::String::init("truncate me", 3, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("tru", str->c_str());
    EXPECT_EQ(3, str->size());
    EXPECT_EQ('\0', str->c_str()[3]); // Verify null terminator
}

// ================================================================================
// Error Handling Tests
// ================================================================================

TEST_F(StringTest, InitWithNullPointerReturnsError) {
    auto result = cslt::String::init(nullptr, 0, allocator);
    
    EXPECT_FALSE(result.hasValue());
    EXPECT_TRUE(result.hasError());
}

TEST_F(StringTest, NullPointerErrorHasDescriptiveMessage) {
    auto result = cslt::String::init(nullptr, 0, allocator);
    
    ASSERT_FALSE(result.hasValue());
    const char* msg = result.error().what();
    EXPECT_NE(nullptr, msg);
    EXPECT_GT(std::strlen(msg), 0);
}

// ================================================================================
// Memory Management Tests
// ================================================================================

TEST_F(StringTest, MultipleStringsFromSameAllocator) {
    auto r1 = cslt::String::init("first", 0, allocator);
    auto r2 = cslt::String::init("second", 0, allocator);
    auto r3 = cslt::String::init("third", 0, allocator);
    
    ASSERT_TRUE(r1.hasValue());
    ASSERT_TRUE(r2.hasValue());
    ASSERT_TRUE(r3.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> s1(r1.value());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> s2(r2.value());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> s3(r3.value());
    
    EXPECT_STREQ("first", s1->c_str());
    EXPECT_STREQ("second", s2->c_str());
    EXPECT_STREQ("third", s3->c_str());
}

TEST_F(StringTest, StringIsProperlyDeletedWhenUniquePtrGoesOutOfScope) {
    // This test verifies no memory leaks occur
    // Run with valgrind or address sanitizer to verify
    {
        auto result = cslt::String::init("temporary", 0, allocator);
        ASSERT_TRUE(result.hasValue());
        cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
        // str goes out of scope here
    }
    // If we reach here without crashes, cleanup worked
    SUCCEED();
}

TEST_F(StringTest, AllocatorPointerIsCorrect) {
    auto result = cslt::String::init("test", 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_EQ(&allocator, str->allocator());
}

// ================================================================================
// Arena Allocator Tests (if dynamic allocation is enabled)
// ================================================================================

#if ARENA_ENABLE_DYNAMIC
TEST_F(StringTest, InitWithArenaAllocator) {
    auto arena_result = cslt::ArenaAllocator::Heap(1024, 0);
    ASSERT_TRUE(arena_result.hasValue());
    
    // arena_result.value() returns a UniquePtr, so we need to use a reference
    auto& arena = arena_result.value();
    
    auto result = cslt::String::init("arena test", 0, *arena);
    ASSERT_TRUE(result.hasValue());
    
    cslt::String* str = result.value();
    EXPECT_STREQ("arena test", str->c_str());
    EXPECT_EQ(10, str->size());
    
    // Verify arena was used
    EXPECT_GT(arena->size(), 0);
}

TEST_F(StringTest, MultipleStringsInArena) {
    auto arena_result = cslt::ArenaAllocator::Heap(2048, 0);
    ASSERT_TRUE(arena_result.hasValue());
    
    // arena_result.value() returns a UniquePtr reference
    auto& arena = arena_result.value();
    
    auto r1 = cslt::String::init("first", 0, *arena);
    auto r2 = cslt::String::init("second", 0, *arena);
    auto r3 = cslt::String::init("third", 0, *arena);
    
    ASSERT_TRUE(r1.hasValue());
    ASSERT_TRUE(r2.hasValue());
    ASSERT_TRUE(r3.hasValue());
    
    cslt::String* s1 = r1.value();
    cslt::String* s2 = r2.value();
    cslt::String* s3 = r3.value();
    
    EXPECT_STREQ("first", s1->c_str());
    EXPECT_STREQ("second", s2->c_str());
    EXPECT_STREQ("third", s3->c_str());
    
    // All three strings should have used arena memory
    size_t expected_min = sizeof(cslt::String) * 3 + 6 + 7 + 6; // struct + strings
    EXPECT_GE(arena->size(), expected_min);
}
#endif

// ================================================================================
// Edge Case Tests
// ================================================================================

TEST_F(StringTest, VeryLargeCapacity) {
    auto result = cslt::String::init("small", 10000, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("small", str->c_str());
    EXPECT_EQ(5, str->size());
    EXPECT_EQ(10001, str->capacity());
}

TEST_F(StringTest, StringWithSpecialCharacters) {
    const char* special = "Hello\nWorld\t!@#$%^&*()";
    auto result = cslt::String::init(special, 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ(special, str->c_str());
}

TEST_F(StringTest, StringWithNullByteInMiddle) {
    // Note: This test shows that strlen-based initialization
    // will only copy up to the first null
    const char embedded_null[] = {'a', 'b', '\0', 'c', 'd', '\0'};
    auto result = cslt::String::init(embedded_null, 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    // Should only contain "ab" because strlen stops at first null
    EXPECT_STREQ("ab", str->c_str());
    EXPECT_EQ(2, str->size());
}

TEST_F(StringTest, CapacityOfOne) {
    auto result = cslt::String::init("abc", 1, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("a", str->c_str());
    EXPECT_EQ(1, str->size());
    EXPECT_EQ(2, str->capacity());
}

TEST_F(StringTest, ZeroCapacityWithEmptyString) {
    auto result = cslt::String::init("", 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_STREQ("", str->c_str());
    EXPECT_EQ(0, str->size());
    EXPECT_EQ(1, str->capacity());
}

// ================================================================================
// Accessor Method Tests
// ================================================================================

TEST_F(StringTest, CStrReturnsValidPointer) {
    auto result = cslt::String::init("test", 0, allocator);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    const char* cstr = str->c_str();
    EXPECT_NE(nullptr, cstr);
    EXPECT_EQ(0, std::strcmp(cstr, "test"));
}

TEST_F(StringTest, SizeReturnsCorrectLength) {
    auto result = cslt::String::init("hello world", 0, allocator);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_EQ(11, str->size());
    EXPECT_EQ(std::strlen(str->c_str()), str->size());
}

TEST_F(StringTest, CapacityIncludesNullTerminator) {
    auto result = cslt::String::init("test", 0, allocator);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_EQ(str->size() + 1, str->capacity());
}

// ================================================================================
// Consistency Tests
// ================================================================================

TEST_F(StringTest, SizeAndCapacityAreConsistent) {
    struct TestCase {
        const char* input;
        size_t capacity;
        size_t expected_size;
        size_t expected_capacity;
    };
    
    TestCase cases[] = {
        {"", 0, 0, 1},
        {"a", 0, 1, 2},
        {"hello", 0, 5, 6},
        {"hi", 10, 2, 11},
        {"truncate", 3, 3, 4},
    };
    
    for (const auto& tc : cases) {
        auto result = cslt::String::init(tc.input, tc.capacity, allocator);
        ASSERT_TRUE(result.hasValue()) << "Failed for input: " << tc.input;
        cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
        
        EXPECT_EQ(tc.expected_size, str->size()) 
            << "Size mismatch for input: " << tc.input;
        EXPECT_EQ(tc.expected_capacity, str->capacity()) 
            << "Capacity mismatch for input: " << tc.input;
    }
}

TEST_F(StringTest, StringIsAlwaysNullTerminated) {
    struct TestCase {
        const char* input;
        size_t capacity;
    };
    
    TestCase cases[] = {
        {"hello", 0},
        {"hello", 10},
        {"hello world", 5},
        {"", 0},
        {"x", 100},
    };
    
    for (const auto& tc : cases) {
        auto result = cslt::String::init(tc.input, tc.capacity, allocator);
        ASSERT_TRUE(result.hasValue()) << "Failed for input: " << tc.input;
        cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
        
        // Verify null terminator
        EXPECT_EQ('\0', str->c_str()[str->size()]) 
            << "Not null-terminated for input: " << tc.input;
    }
}

// ================================================================================
// Stress Tests
// ================================================================================

TEST_F(StringTest, ManySmallStrings) {
    const int count = 1000;
    std::vector<cslt::UniquePtr<cslt::String, cslt::StringDeleter>> strings;
    
    for (int i = 0; i < count; ++i) {
        auto result = cslt::String::init("x", 0, allocator);
        ASSERT_TRUE(result.hasValue()) << "Failed at iteration " << i;
        strings.emplace_back(result.value());
    }
    
    EXPECT_EQ(count, strings.size());
    
    // Verify all strings are still valid
    for (const auto& str : strings) {
        EXPECT_STREQ("x", str->c_str());
    }
}

TEST_F(StringTest, LargeStringAllocation) {
    // Create a 10KB string
    std::string large_input(10000, 'a');
    auto result = cslt::String::init(large_input.c_str(), 0, allocator);
    
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
    
    EXPECT_EQ(10000, str->size());
    EXPECT_EQ(10001, str->capacity());
    EXPECT_EQ('a', str->c_str()[0]);
    EXPECT_EQ('a', str->c_str()[9999]);
}
// ================================================================================
// ================================================================================
// eof
// ================================================================================
// ================================================================================
// eof
