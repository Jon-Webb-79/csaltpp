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
// Test Fixture
// ================================================================================

class StringConcatTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;  // Adjust based on your actual allocator type
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
};

// ================================================================================
// Basic Functionality Tests
// ================================================================================

TEST_F(StringConcatTest, ConcatCStringBasic) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    bool success = str->concat(" world");
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello world");
    EXPECT_EQ(str->size(), 11u);
}

TEST_F(StringConcatTest, ConcatStringObjectBasic) {
    auto str1 = makeString("hello");
    auto str2 = makeString(" world");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    bool success = str1->concat(*str2);
    EXPECT_TRUE(success);
    EXPECT_STREQ(str1->c_str(), "hello world");
    EXPECT_EQ(str1->size(), 11u);
}

TEST_F(StringConcatTest, ConcatEmptyString) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    size_t original_len = str->size();
    bool success = str->concat("");
    
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello");
    EXPECT_EQ(str->size(), original_len);
}

TEST_F(StringConcatTest, ConcatToEmptyString) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    bool success = str->concat("hello");
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello");
    EXPECT_EQ(str->size(), 5u);
}

TEST_F(StringConcatTest, MultipleConcatenations) {
    auto str = makeString("a");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_TRUE(str->concat("b"));
    EXPECT_STREQ(str->c_str(), "ab");
    
    EXPECT_TRUE(str->concat("c"));
    EXPECT_STREQ(str->c_str(), "abc");
    
    EXPECT_TRUE(str->concat("d"));
    EXPECT_STREQ(str->c_str(), "abcd");
    
    EXPECT_EQ(str->size(), 4u);
}

// ================================================================================
// Capacity and Growth Tests
// ================================================================================

TEST_F(StringConcatTest, ConcatWithSufficientCapacity) {
    // Create string with extra capacity
    auto str = makeString("hello", 20);
    ASSERT_NE(str.get(), nullptr);
    
    size_t original_capacity = str->capacity();
    
    bool success = str->concat(" world");
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello world");
    
    // Capacity should not have changed
    EXPECT_EQ(str->capacity(), original_capacity);
}

TEST_F(StringConcatTest, ConcatRequiresGrowth) {
    // Create string with exact capacity (no room for growth)
    auto str = makeString("hello", 0);
    ASSERT_NE(str.get(), nullptr);
    
    size_t original_capacity = str->capacity();
    
    bool success = str->concat(" world");
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello world");
    
    // Capacity should have grown
    EXPECT_GT(str->capacity(), original_capacity);
}

TEST_F(StringConcatTest, ConcatLargeString) {
    auto str = makeString("start");
    ASSERT_NE(str.get(), nullptr);
    
    // Create a large string to append
    std::string large_append(1000, 'x');
    
    bool success = str->concat(large_append.c_str());
    EXPECT_TRUE(success);
    EXPECT_EQ(str->size(), 5u + 1000u);
    
    // Verify content
    EXPECT_EQ(str->c_str()[0], 's');
    EXPECT_EQ(str->c_str()[5], 'x');
    EXPECT_EQ(str->c_str()[1004], 'x');
}

// ================================================================================
// Edge Cases and Error Handling
// ================================================================================

TEST_F(StringConcatTest, ConcatNullCString) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    bool success = str->concat(nullptr);
    EXPECT_FALSE(success);
    
    // Original string should be unchanged
    EXPECT_STREQ(str->c_str(), "hello");
    EXPECT_EQ(str->size(), 5u);
}

TEST_F(StringConcatTest, ConcatSelfAlias) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Concatenate the string with itself (aliasing case)
    bool success = str->concat(str->c_str());
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hellohello");
    EXPECT_EQ(str->size(), 10u);
}

TEST_F(StringConcatTest, ConcatSelfSubstring) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    // Point to a substring within the buffer
    const char* substr = str->c_str() + 6;  // "world"
    
    bool success = str->concat(substr);
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "hello worldworld");
    EXPECT_EQ(str->size(), 16u);
}

TEST_F(StringConcatTest, ConcatWithSpecialCharacters) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    bool success = str->concat("\t\n\r");
    EXPECT_TRUE(success);
    EXPECT_EQ(str->size(), 8u);
    
    // Verify special characters are present
    EXPECT_EQ(str->c_str()[5], '\t');
    EXPECT_EQ(str->c_str()[6], '\n');
    EXPECT_EQ(str->c_str()[7], '\r');
}

TEST_F(StringConcatTest, ConcatUnicode) {
    auto str = makeString("Hello ");
    ASSERT_NE(str.get(), nullptr);
    
    bool success = str->concat("世界");  // "world" in Chinese
    EXPECT_TRUE(success);
    EXPECT_STREQ(str->c_str(), "Hello 世界");
}

// ================================================================================
// String-to-String Concatenation Tests
// ================================================================================

TEST_F(StringConcatTest, ConcatTwoStrings) {
    auto str1 = makeString("foo");
    auto str2 = makeString("bar");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    bool success = str1->concat(*str2);
    EXPECT_TRUE(success);
    EXPECT_STREQ(str1->c_str(), "foobar");
    
    // str2 should be unchanged
    EXPECT_STREQ(str2->c_str(), "bar");
}

TEST_F(StringConcatTest, ConcatEmptyStringObject) {
    auto str1 = makeString("hello");
    auto str2 = makeString("");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    bool success = str1->concat(*str2);
    EXPECT_TRUE(success);
    EXPECT_STREQ(str1->c_str(), "hello");
}

TEST_F(StringConcatTest, ConcatMultipleStringObjects) {
    auto str1 = makeString("a");
    auto str2 = makeString("b");
    auto str3 = makeString("c");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    ASSERT_NE(str3.get(), nullptr);
    
    EXPECT_TRUE(str1->concat(*str2));
    EXPECT_TRUE(str1->concat(*str3));
    EXPECT_STREQ(str1->c_str(), "abc");
}

// ================================================================================
// Boundary and Stress Tests
// ================================================================================

TEST_F(StringConcatTest, ConcatMaxLengthString) {
    auto str = makeString("a");
    ASSERT_NE(str.get(), nullptr);
    
    // Concatenate many times to test growth
    for (int i = 0; i < 100; ++i) {
        bool success = str->concat("x");
        EXPECT_TRUE(success);
    }
    
    EXPECT_EQ(str->size(), 101u);
    EXPECT_EQ(str->c_str()[0], 'a');
    EXPECT_EQ(str->c_str()[100], 'x');
}

TEST_F(StringConcatTest, ConcatAlternatingTypes) {
    auto str1 = makeString("start");
    auto str2 = makeString("-mid-");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    EXPECT_TRUE(str1->concat(*str2));      // String object
    EXPECT_TRUE(str1->concat("end"));      // C-string
    
    EXPECT_STREQ(str1->c_str(), "start-mid-end");
}

// ================================================================================
// Null Terminator Tests
// ================================================================================

TEST_F(StringConcatTest, NullTerminatorMaintained) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    str->concat(" world");
    
    // Verify null terminator is present
    EXPECT_EQ(str->c_str()[str->size()], '\0');
    
    // Verify strlen matches size
    EXPECT_EQ(std::strlen(str->c_str()), str->size());
}

TEST_F(StringConcatTest, MultipleConcat_NullTerminator) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    for (int i = 0; i < 10; ++i) {
        str->concat("a");
        EXPECT_EQ(str->c_str()[str->size()], '\0');
    }
}

// ================================================================================
// Memory Management Tests
// ================================================================================

TEST_F(StringConcatTest, ConcatDoesNotLeakMemory) {
    // This test relies on your allocator tracking allocations
    // Adjust based on your allocator's API
    
    auto str = makeString("test");
    ASSERT_NE(str.get(), nullptr);
    
    // Perform operations that trigger reallocation
    for (int i = 0; i < 50; ++i) {
        str->concat("x");
    }
    
    // When str goes out of scope, StringDeleter should clean up
    // If your allocator tracks allocations, you can verify here
}

// ================================================================================
// Integration Tests
// ================================================================================

TEST_F(StringConcatTest, BuildSentence) {
    auto str = makeString("The");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_TRUE(str->concat(" quick"));
    EXPECT_TRUE(str->concat(" brown"));
    EXPECT_TRUE(str->concat(" fox"));
    
    EXPECT_STREQ(str->c_str(), "The quick brown fox");
    EXPECT_EQ(str->size(), 19u);
}

TEST_F(StringConcatTest, BuildPath) {
    auto path = makeString("/usr");
    ASSERT_NE(path.get(), nullptr);
    
    EXPECT_TRUE(path->concat("/local"));
    EXPECT_TRUE(path->concat("/bin"));
    
    EXPECT_STREQ(path->c_str(), "/usr/local/bin");
}

// ================================================================================
// Performance Hints Tests (Optional)
// ================================================================================

TEST_F(StringConcatTest, PreallocatedCapacityPerformance) {
    // Test that pre-allocating capacity reduces reallocations
    
    auto str1 = makeString("", 100);  // Pre-allocated
    auto str2 = makeString("", 0);     // Minimal allocation
    
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    size_t capacity1_before = str1->capacity();
    size_t capacity2_before = str2->capacity();
    
    for (int i = 0; i < 10; ++i) {
        str1->concat("1234567890");
        str2->concat("1234567890");
    }
    
    // str1 should not have reallocated (or reallocated less)
    EXPECT_EQ(str1->capacity(), capacity1_before);
    
    // str2 likely reallocated multiple times
    EXPECT_GT(str2->capacity(), capacity2_before);
}
// ================================================================================ 
// ================================================================================ 


// #include <gtest/gtest.h>
// #include "string.hpp"
// #include "allocator.hpp"
// #include <vector>
// #include <algorithm>
// #include <string>
#include <random>

// ================================================================================
// Test Fixture
// ================================================================================

class StringCompareExtendedTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
    
    // Helper to create a string from std::string
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const std::string& str, size_t capacity = 0) {
        return makeString(str.c_str(), capacity);
    }
};

// ================================================================================
// SIMD Boundary Tests (16, 32, 64 byte boundaries)
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_Exactly16Bytes_Equal) {
    std::string str16(16, 'x');
    
    auto s1 = makeString(str16);
    auto s2 = makeString(str16);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_Exactly16Bytes_DifferAtEnd) {
    std::string str1(16, 'x');
    std::string str2 = str1;
    str2[15] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);  // 'x' < 'y'
}

TEST_F(StringCompareExtendedTest, Compare_Exactly32Bytes_Equal) {
    std::string str32(32, 'a');
    
    auto s1 = makeString(str32);
    auto s2 = makeString(str32);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_Exactly32Bytes_DifferAtMiddle) {
    std::string str1(32, 'a');
    std::string str2 = str1;
    str2[16] = 'b';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);  // 'a' < 'b'
}

TEST_F(StringCompareExtendedTest, Compare_Exactly64Bytes_Equal) {
    std::string str64(64, 'z');
    
    auto s1 = makeString(str64);
    auto s2 = makeString(str64);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_Exactly64Bytes_DifferAtEnd) {
    std::string str1(64, 'z');
    std::string str2 = str1;
    str2[63] = 'a';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 1);  // 'z' > 'a'
}

// ================================================================================
// Tests Crossing SIMD Boundaries
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_15Bytes_JustUnder16) {
    std::string str(15, 'x');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_17Bytes_JustOver16) {
    std::string str(17, 'x');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_31Bytes_JustUnder32) {
    std::string str(31, 'a');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_33Bytes_JustOver32) {
    std::string str(33, 'a');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_65Bytes_JustOver64) {
    std::string str(65, 'z');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

// ================================================================================
// Early Difference Detection Tests
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte0) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[0] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte1) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[1] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte15) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[15] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte16) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[16] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte31) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[31] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte32) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[32] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte63) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[63] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_DifferAtByte64) {
    std::string str1(100, 'x');
    std::string str2 = str1;
    str2[64] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

// ================================================================================
// Large String Tests (SIMD Performance)
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_1KB_Equal) {
    std::string str(1024, 'x');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_1KB_DifferAtEnd) {
    std::string str1(1024, 'x');
    std::string str2 = str1;
    str2[1023] = 'y';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

TEST_F(StringCompareExtendedTest, Compare_10KB_Equal) {
    std::string str(10240, 'a');
    
    auto s1 = makeString(str);
    auto s2 = makeString(str);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_10KB_DifferAtMiddle) {
    std::string str1(10240, 'a');
    std::string str2 = str1;
    str2[5120] = 'b';
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}

// ================================================================================
// All Byte Values Tests (0-255)
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_AllByteValues) {
    // Create strings with all possible byte values
    std::string str1;
    std::string str2;
    
    for (int i = 1; i < 256; ++i) {  // Skip 0 (null terminator)
        str1.push_back(static_cast<char>(i));
        str2.push_back(static_cast<char>(i));
    }
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_HighByteValues) {
    // Test with high byte values (128-255)
    std::string str1(100, static_cast<char>(200));
    std::string str2(100, static_cast<char>(201));
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);  // 200 < 201
}

// ================================================================================
// Random Data Tests
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_RandomData_LargeStrings) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(1, 255);
    
    std::string str1;
    std::string str2;
    
    // Create 1KB random strings
    for (int i = 0; i < 1024; ++i) {
        char c = static_cast<char>(dis(gen));
        str1.push_back(c);
        str2.push_back(c);
    }
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_RandomData_DifferAtRandomPosition) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 255);
    
    std::string str1;
    std::string str2;
    
    for (int i = 0; i < 1024; ++i) {
        char c = static_cast<char>(dis(gen));
        str1.push_back(c);
        str2.push_back(c);
    }
    
    // Change byte at position 500
    str2[500] = static_cast<char>((static_cast<unsigned char>(str2[500]) + 1) % 256);
    if (str2[500] == 0) str2[500] = 1;  // Avoid null
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_NE(cmp, 0);  // Should differ
}

// ================================================================================
// Alignment Tests (Unaligned Data)
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_UnalignedData_Offset1) {
    std::string base(100, 'x');
    std::string str1 = " " + base;  // Offset by 1
    std::string str2 = " " + base;
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_UnalignedData_Offset7) {
    std::string base(100, 'x');
    std::string str1 = "1234567" + base;  // Offset by 7
    std::string str2 = "1234567" + base;
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_UnalignedData_Offset15) {
    std::string base(100, 'x');
    std::string str1 = "123456789012345" + base;  // Offset by 15
    std::string str2 = "123456789012345" + base;
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

// ================================================================================
// Repeated Pattern Tests
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_RepeatingPattern_2Byte) {
    std::string str1;
    std::string str2;
    
    for (int i = 0; i < 500; ++i) {
        str1 += "ab";
        str2 += "ab";
    }
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_RepeatingPattern_4Byte) {
    std::string str1;
    std::string str2;
    
    for (int i = 0; i < 250; ++i) {
        str1 += "test";
        str2 += "test";
    }
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, 0);
}

TEST_F(StringCompareExtendedTest, Compare_RepeatingPattern_BreakAtEnd) {
    std::string str1;
    std::string str2;
    
    for (int i = 0; i < 100; ++i) {
        str1 += "pattern";
        str2 += "pattern";
    }
    
    str1 += "end1";
    str2 += "end2";
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);  // "end1" < "end2"
}

// ================================================================================
// Length Mismatch Tests
// ================================================================================

TEST_F(StringCompareExtendedTest, Compare_LengthDiff_BothLarge) {
    std::string str1(1000, 'x');
    std::string str2(1001, 'x');
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);  // Shorter is less
}

TEST_F(StringCompareExtendedTest, Compare_LengthDiff_AcrossSIMDBoundary) {
    std::string str1(63, 'x');
    std::string str2(65, 'x');
    
    auto s1 = makeString(str1);
    auto s2 = makeString(str2);
    ASSERT_NE(s1.get(), nullptr);
    ASSERT_NE(s2.get(), nullptr);
    
    int8_t cmp = s1->compare(*s2);
    EXPECT_EQ(cmp, -1);
}
// -------------------------------------------------------------------------------- 


class StringResetTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
};

// ================================================================================
// Basic Functionality Tests
// ================================================================================

TEST_F(StringResetTest, Reset_NonEmptyString) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_EQ(str->size(), 11u);
    EXPECT_STREQ(str->c_str(), "hello world");
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
    EXPECT_EQ(str->c_str()[0], '\0');
}

TEST_F(StringResetTest, Reset_EmptyString) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_EQ(str->size(), 0u);
    
    str->reset();  // Should be safe on empty string
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
}

TEST_F(StringResetTest, Reset_SingleCharacter) {
    auto str = makeString("x");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_EQ(str->size(), 1u);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
}

TEST_F(StringResetTest, Reset_LongString) {
    std::string long_str(1000, 'x');
    auto str = makeString(long_str.c_str());
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_EQ(str->size(), 1000u);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
}

// ================================================================================
// Capacity Preservation Tests
// ================================================================================

TEST_F(StringResetTest, Reset_PreservesCapacity) {
    auto str = makeString("hello", 100);  // Pre-allocate 100 bytes
    ASSERT_NE(str.get(), nullptr);
    
    size_t original_capacity = str->capacity();
    EXPECT_EQ(original_capacity, 101u);  // 100 + null terminator
    
    str->reset();
    
    EXPECT_EQ(str->capacity(), original_capacity);
}

TEST_F(StringResetTest, Reset_NoReallocation) {
    auto str = makeString("initial", 100);
    ASSERT_NE(str.get(), nullptr);
    
    const char* buffer_ptr = str->c_str();
    size_t capacity = str->capacity();
    
    str->reset();
    
    // Buffer pointer should be the same
    EXPECT_EQ(str->c_str(), buffer_ptr);
    EXPECT_EQ(str->capacity(), capacity);
}

TEST_F(StringResetTest, Reset_PreservesAllocator) {
    auto str = makeString("test", 100);
    ASSERT_NE(str.get(), nullptr);
    
    cslt::Allocator* alloc_before = str->allocator();
    
    str->reset();
    
    cslt::Allocator* alloc_after = str->allocator();
    EXPECT_EQ(alloc_before, alloc_after);
}

// ================================================================================
// Buffer Reuse Tests
// ================================================================================

TEST_F(StringResetTest, Reset_AllowsReuse) {
    auto str = makeString("", 100);
    ASSERT_NE(str.get(), nullptr);
    
    str->concat("first message");
    EXPECT_STREQ(str->c_str(), "first message");
    
    str->reset();
    EXPECT_STREQ(str->c_str(), "");
    
    str->concat("second message");
    EXPECT_STREQ(str->c_str(), "second message");
}

TEST_F(StringResetTest, MultipleResetReuse) {
    auto str = makeString("", 100);
    ASSERT_NE(str.get(), nullptr);
    
    for (int i = 0; i < 10; ++i) {
        str->reset();
        str->concat("iteration ");
        str->concat(std::to_string(i).c_str());
        
        EXPECT_GT(str->size(), 0u);
    }
    
    // Should still have original capacity
    EXPECT_EQ(str->capacity(), 101u);
}

// ================================================================================
// Multiple Reset Tests
// ================================================================================

TEST_F(StringResetTest, MultipleResets) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    EXPECT_EQ(str->size(), 0u);
    
    str->reset();
    EXPECT_EQ(str->size(), 0u);
    
    str->reset();
    EXPECT_EQ(str->size(), 0u);
}

// ================================================================================
// Interaction with Other Methods
// ================================================================================

TEST_F(StringResetTest, Reset_ThenConcat) {
    auto str = makeString("old data");
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    str->concat("new data");
    
    EXPECT_STREQ(str->c_str(), "new data");
    EXPECT_EQ(str->size(), 8u);
}

TEST_F(StringResetTest, Concat_ThenReset_ThenConcat) {
    auto str = makeString("", 100);
    ASSERT_NE(str.get(), nullptr);
    
    str->concat("first");
    EXPECT_STREQ(str->c_str(), "first");
    
    str->reset();
    EXPECT_STREQ(str->c_str(), "");
    
    str->concat("second");
    EXPECT_STREQ(str->c_str(), "second");
}

TEST_F(StringResetTest, Reset_ThenCompare) {
    auto str1 = makeString("hello");
    auto str2 = makeString("");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    str1->reset();
    
    int8_t cmp = str1->compare(*str2);
    EXPECT_EQ(cmp, 0);  // Both should be empty
}

TEST_F(StringResetTest, Reset_ThenSize) {
    auto str = makeString("test");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_GT(str->size(), 0u);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
}

// ================================================================================
// Null Terminator Tests
// ================================================================================

TEST_F(StringResetTest, Reset_EnsuresNullTerminator) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    
    EXPECT_EQ(str->c_str()[0], '\0');
    EXPECT_EQ(std::strlen(str->c_str()), 0u);
}

// ================================================================================
// Performance Pattern Tests
// ================================================================================

TEST_F(StringResetTest, LoopReusePattern) {
    auto buffer = makeString("", 1000);
    ASSERT_NE(buffer.get(), nullptr);
    
    size_t original_capacity = buffer->capacity();
    
    for (int i = 0; i < 100; ++i) {
        buffer->reset();
        
        buffer->concat("Iteration ");
        buffer->concat(std::to_string(i).c_str());
        
        // Verify contents
        EXPECT_GT(buffer->size(), 0u);
    }
    
    // Should not have reallocated
    EXPECT_EQ(buffer->capacity(), original_capacity);
}

TEST_F(StringResetTest, BuildProcessResetPattern) {
    auto str = makeString("", 500);
    ASSERT_NE(str.get(), nullptr);
    
    // Pattern: build, process, reset
    for (int i = 0; i < 50; ++i) {
        // Build
        str->concat("data_");
        str->concat(std::to_string(i).c_str());
        
        // Process (verify)
        EXPECT_GT(str->size(), 0u);
        
        // Reset for next iteration
        str->reset();
        EXPECT_EQ(str->size(), 0u);
    }
}

// ================================================================================
// Special Characters Tests
// ================================================================================

TEST_F(StringResetTest, Reset_WithSpecialCharacters) {
    auto str = makeString("hello\nworld\t!");
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
}

TEST_F(StringResetTest, Reset_WithHighByteValues) {
    std::string data;
    data.push_back(static_cast<char>(200));
    data.push_back(static_cast<char>(201));
    data.push_back(static_cast<char>(202));
    
    auto str = makeString(data.c_str());
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_EQ(str->c_str()[0], '\0');
}

// ================================================================================
// Edge Cases
// ================================================================================

TEST_F(StringResetTest, Reset_VeryLongString) {
    std::string long_str(10000, 'x');
    auto str = makeString(long_str.c_str());
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_EQ(str->size(), 10000u);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_STREQ(str->c_str(), "");
}

TEST_F(StringResetTest, Reset_MinimalCapacity) {
    auto str = makeString("x", 0);  // Minimal allocation
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_GT(str->capacity(), 0u);  // Should still have at least 1 for null
}

TEST_F(StringResetTest, Reset_MaximalCapacity) {
    auto str = makeString("", 10000);  // Large pre-allocation
    ASSERT_NE(str.get(), nullptr);
    
    str->concat("small");
    str->reset();
    
    EXPECT_EQ(str->size(), 0u);
    EXPECT_EQ(str->capacity(), 10001u);  // Should preserve large capacity
}

// ================================================================================
// Integration Tests
// ================================================================================

TEST_F(StringResetTest, ResetInStringBuilder) {
    auto builder = makeString("", 1000);
    ASSERT_NE(builder.get(), nullptr);
    
    // Build first message
    builder->concat("Hello, ");
    builder->concat("World!");
    EXPECT_STREQ(builder->c_str(), "Hello, World!");
    
    // Reset and build second message
    builder->reset();
    builder->concat("Goodbye, ");
    builder->concat("World!");
    EXPECT_STREQ(builder->c_str(), "Goodbye, World!");
}

TEST_F(StringResetTest, ResetInDataProcessing) {
    auto data = makeString("", 200);
    ASSERT_NE(data.get(), nullptr);
    
    std::vector<std::string> results;
    
    for (int i = 0; i < 10; ++i) {
        data->reset();
        data->concat("Record_");
        data->concat(std::to_string(i).c_str());
        
        // Save result
        results.push_back(std::string(data->c_str()));
    }
    
    EXPECT_EQ(results.size(), 10u);
    EXPECT_EQ(results[0], "Record_0");
    EXPECT_EQ(results[9], "Record_9");
}
// -------------------------------------------------------------------------------- 

class StringCopyOverloadTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
};

// ================================================================================
// Basic Copy Tests - No Argument Version (Uses Same Allocator)
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_NoArg_BasicString) {
    auto original = makeString("hello world");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "hello world");
    EXPECT_EQ(copy->size(), 11u);
}

TEST_F(StringCopyOverloadTest, Copy_NoArg_EmptyString) {
    auto original = makeString("");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "");
    EXPECT_EQ(copy->size(), 0u);
}

TEST_F(StringCopyOverloadTest, Copy_NoArg_SingleCharacter) {
    auto original = makeString("x");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "x");
    EXPECT_EQ(copy->size(), 1u);
}

TEST_F(StringCopyOverloadTest, Copy_NoArg_LongString) {
    std::string long_str(1000, 'x');
    auto original = makeString(long_str.c_str());
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), original->c_str());
    EXPECT_EQ(copy->size(), 1000u);
}

// ================================================================================
// Basic Copy Tests - With Allocator Argument
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_WithAllocator_BasicString) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    
    auto r = cslt::String::init("hello world", 0, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    auto copy_r = original->copy(alloc2);
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "hello world");
    EXPECT_EQ(copy->size(), 11u);
}

TEST_F(StringCopyOverloadTest, Copy_WithAllocator_EmptyString) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    
    auto r = cslt::String::init("", 0, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    auto copy_r = original->copy(alloc2);
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "");
    EXPECT_EQ(copy->size(), 0u);
}

// ================================================================================
// Independence Tests (Deep Copy Verification)
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_NoArg_IsIndependent) {
    auto original = makeString("original");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Buffers should be different
    EXPECT_NE(original->c_str(), copy->c_str());
    
    // But content should match
    EXPECT_STREQ(original->c_str(), copy->c_str());
}

TEST_F(StringCopyOverloadTest, Copy_NoArg_ModifyOriginal_CopyUnchanged) {
    auto original = makeString("hello", 100);
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Modify original
    original->concat(" world");
    
    // Copy should be unchanged
    EXPECT_STREQ(copy->c_str(), "hello");
    EXPECT_EQ(copy->size(), 5u);
    
    // Original should be modified
    EXPECT_STREQ(original->c_str(), "hello world");
}

TEST_F(StringCopyOverloadTest, Copy_NoArg_ModifyCopy_OriginalUnchanged) {
    auto original = makeString("hello");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Modify copy
    copy->concat(" world");
    
    // Original should be unchanged
    EXPECT_STREQ(original->c_str(), "hello");
    EXPECT_EQ(original->size(), 5u);
    
    // Copy should be modified
    EXPECT_STREQ(copy->c_str(), "hello world");
}

TEST_F(StringCopyOverloadTest, Copy_WithAllocator_IsIndependent) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    
    auto r = cslt::String::init("test", 0, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    auto copy_r = original->copy(alloc2);
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Buffers should be different
    EXPECT_NE(original->c_str(), copy->c_str());
    
    // Content should match
    EXPECT_STREQ(original->c_str(), copy->c_str());
}

// ================================================================================
// Allocator Tests
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_NoArg_SameAllocator) {
    auto original = makeString("test");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Should use same allocator
    EXPECT_EQ(original->allocator(), copy->allocator());
}

TEST_F(StringCopyOverloadTest, Copy_WithAllocator_DifferentAllocator) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    
    auto r = cslt::String::init("test", 0, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    auto copy_r = original->copy(alloc2);
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Should use different allocator
    EXPECT_NE(original->allocator(), copy->allocator());
    EXPECT_EQ(copy->allocator(), &alloc2);
}

// ================================================================================
// Multiple Copy Tests
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_NoArg_MultipleCopies) {
    auto original = makeString("original");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy1_r = original->copy();
    auto copy2_r = original->copy();
    auto copy3_r = original->copy();
    
    ASSERT_TRUE(copy1_r.hasValue());
    ASSERT_TRUE(copy2_r.hasValue());
    ASSERT_TRUE(copy3_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy1(copy1_r.value());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy2(copy2_r.value());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy3(copy3_r.value());
    
    // All copies should match original
    EXPECT_STREQ(copy1->c_str(), "original");
    EXPECT_STREQ(copy2->c_str(), "original");
    EXPECT_STREQ(copy3->c_str(), "original");
    
    // All should be independent
    EXPECT_NE(copy1->c_str(), copy2->c_str());
    EXPECT_NE(copy2->c_str(), copy3->c_str());
    EXPECT_NE(copy1->c_str(), copy3->c_str());
}

TEST_F(StringCopyOverloadTest, Copy_CopyOfCopy) {
    auto original = makeString("test");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy1_r = original->copy();
    ASSERT_TRUE(copy1_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy1(copy1_r.value());
    
    auto copy2_r = copy1->copy();
    ASSERT_TRUE(copy2_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy2(copy2_r.value());
    
    // Copy of copy should match original
    EXPECT_STREQ(copy2->c_str(), "test");
    EXPECT_EQ(copy2->size(), 4u);
}

TEST_F(StringCopyOverloadTest, Copy_MixedAllocators_MultipleCopies) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    cslt::HeapAllocator alloc3;
    
    auto r = cslt::String::init("test", 0, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    // Copy with same allocator
    auto copy1_r = original->copy();
    ASSERT_TRUE(copy1_r.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy1(copy1_r.value());
    
    // Copy with different allocator
    auto copy2_r = original->copy(alloc2);
    ASSERT_TRUE(copy2_r.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy2(copy2_r.value());
    
    // Another copy with different allocator
    auto copy3_r = original->copy(alloc3);
    ASSERT_TRUE(copy3_r.hasValue());
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy3(copy3_r.value());
    
    // All should have same content
    EXPECT_STREQ(copy1->c_str(), "test");
    EXPECT_STREQ(copy2->c_str(), "test");
    EXPECT_STREQ(copy3->c_str(), "test");
    
    // But different allocators
    EXPECT_EQ(copy1->allocator(), &alloc1);
    EXPECT_EQ(copy2->allocator(), &alloc2);
    EXPECT_EQ(copy3->allocator(), &alloc3);
}

// ================================================================================
// Capacity Tests
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_NoArg_CapacityMatchesLength) {
    auto original = makeString("hello", 100);  // Large capacity
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Copy capacity should match length, not original capacity
    EXPECT_EQ(copy->capacity(), original->size() + 1);  // len + null
    EXPECT_LT(copy->capacity(), original->capacity());
}

TEST_F(StringCopyOverloadTest, Copy_WithAllocator_CapacityMatchesLength) {
    cslt::HeapAllocator alloc1;
    cslt::HeapAllocator alloc2;
    
    auto r = cslt::String::init("hello", 100, alloc1);
    ASSERT_TRUE(r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
    
    auto copy_r = original->copy(alloc2);
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Copy capacity should match length, not original capacity
    EXPECT_EQ(copy->capacity(), original->size() + 1);
    EXPECT_LT(copy->capacity(), original->capacity());
}

// ================================================================================
// Special Content Tests
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_SpecialCharacters) {
    auto original = makeString("hello\nworld\t!");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "hello\nworld\t!");
}

TEST_F(StringCopyOverloadTest, Copy_HighByteValues) {
    std::string data;
    data.push_back(static_cast<char>(200));
    data.push_back(static_cast<char>(201));
    data.push_back(static_cast<char>(202));
    
    auto original = makeString(data.c_str());
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_EQ(copy->size(), 3u);
    EXPECT_EQ(static_cast<unsigned char>(copy->c_str()[0]), 200);
    EXPECT_EQ(static_cast<unsigned char>(copy->c_str()[1]), 201);
    EXPECT_EQ(static_cast<unsigned char>(copy->c_str()[2]), 202);
}

TEST_F(StringCopyOverloadTest, Copy_UnicodeString) {
    auto original = makeString("Hello 世界");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "Hello 世界");
}

// ================================================================================
// Integration with Other Methods
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_ThenConcat) {
    auto original = makeString("hello");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    copy->concat(" world");
    
    EXPECT_STREQ(copy->c_str(), "hello world");
    EXPECT_STREQ(original->c_str(), "hello");
}

TEST_F(StringCopyOverloadTest, Copy_ThenCompare) {
    auto original = makeString("test");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    int8_t cmp = copy->compare(*original);
    EXPECT_EQ(cmp, 0);  // Should be equal
}

TEST_F(StringCopyOverloadTest, Copy_ThenReset) {
    auto original = makeString("data");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    copy->reset();
    
    EXPECT_STREQ(copy->c_str(), "");
    EXPECT_STREQ(original->c_str(), "data");
}

// ================================================================================
// Use Case Tests
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_BackupPattern) {
    auto data = makeString("important data", 100);
    ASSERT_NE(data.get(), nullptr);
    
    // Create backup using same allocator
    auto backup_r = data->copy();
    ASSERT_TRUE(backup_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> backup(backup_r.value());
    
    // Modify data
    data->concat(" - modified");
    
    // Can restore from backup
    data->reset();
    data->concat(backup->c_str());
    
    EXPECT_STREQ(data->c_str(), "important data");
}

TEST_F(StringCopyOverloadTest, Copy_VersionHistory) {
    auto str = makeString("version 1", 100);
    ASSERT_NE(str.get(), nullptr);
    
    std::vector<cslt::UniquePtr<cslt::String, cslt::StringDeleter>> versions;
    
    // Save version 1
    auto v1_r = str->copy();
    ASSERT_TRUE(v1_r.hasValue());
    versions.push_back(cslt::UniquePtr<cslt::String, cslt::StringDeleter>(v1_r.value()));
    
    // Modify to version 2
    str->concat(" updated");
    auto v2_r = str->copy();
    ASSERT_TRUE(v2_r.hasValue());
    versions.push_back(cslt::UniquePtr<cslt::String, cslt::StringDeleter>(v2_r.value()));
    
    // Modify to version 3
    str->concat(" again");
    
    // Verify versions
    EXPECT_STREQ(versions[0]->c_str(), "version 1");
    EXPECT_STREQ(versions[1]->c_str(), "version 1 updated");
    EXPECT_STREQ(str->c_str(), "version 1 updated again");
}

// ================================================================================
// Edge Cases
// ================================================================================

TEST_F(StringCopyOverloadTest, Copy_VeryLongString) {
    std::string long_str(10000, 'x');
    auto original = makeString(long_str.c_str());
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_EQ(copy->size(), 10000u);
    EXPECT_STREQ(copy->c_str(), original->c_str());
}

TEST_F(StringCopyOverloadTest, Copy_AfterReset) {
    auto str = makeString("initial", 100);
    ASSERT_NE(str.get(), nullptr);
    
    str->reset();
    str->concat("new");
    
    auto copy_r = str->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "new");
}

TEST_F(StringCopyOverloadTest, Copy_AfterMultipleOperations) {
    auto str = makeString("", 100);
    ASSERT_NE(str.get(), nullptr);
    
    str->concat("hello");
    str->concat(" world");
    str->reset();
    str->concat("final");
    
    auto copy_r = str->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    EXPECT_STREQ(copy->c_str(), "final");
}
// -------------------------------------------------------------------------------- 

class StringIsPtrTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
};

// ================================================================================
// Basic Pointer Tests - is_ptr(const void*)
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_StartOfBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    
    EXPECT_TRUE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_MiddleOfBuffer) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 6;  // Points to 'w' in "world"
    
    EXPECT_TRUE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_EndOfString) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + str->size();  // Points to null terminator
    
    EXPECT_TRUE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_LastByteOfBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Point to last byte before one-past-end
    const void* ptr = str->c_str() + (str->capacity() - 1);
    
    EXPECT_TRUE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_ExternalPointer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const char* external = "external string";
    
    EXPECT_FALSE(str->is_ptr(external));
}

TEST_F(StringIsPtrTest, IsPtr_NullPointer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_FALSE(str->is_ptr(nullptr));
}

TEST_F(StringIsPtrTest, IsPtr_BeforeBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Create pointer before buffer
    const void* ptr = str->c_str() - 1;
    
    EXPECT_FALSE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_OnePastEnd) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // One-past-end should NOT be contained
    const void* ptr = str->c_str() + str->capacity();
    
    EXPECT_FALSE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_WayAfterBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Point way beyond allocated buffer
    const void* ptr = str->c_str() + str->capacity() + 100;
    
    EXPECT_FALSE(str->is_ptr(ptr));
}

// ================================================================================
// Basic Range Tests - is_ptr(const void*, size_t)
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_Range_EntireString) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_TRUE(str->is_ptr(str->c_str(), str->size()));
}

TEST_F(StringIsPtrTest, IsPtr_Range_EntireBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_TRUE(str->is_ptr(str->c_str(), str->capacity()));
}

TEST_F(StringIsPtrTest, IsPtr_Range_SingleByte) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 2;
    
    EXPECT_TRUE(str->is_ptr(ptr, 1));
}

TEST_F(StringIsPtrTest, IsPtr_Range_Substring) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 6;  // "world"
    
    EXPECT_TRUE(str->is_ptr(ptr, 5));  // 5 bytes for "world"
}

TEST_F(StringIsPtrTest, IsPtr_Range_ZeroBytes) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    
    EXPECT_FALSE(str->is_ptr(ptr, 0));  // Zero bytes is invalid
}

TEST_F(StringIsPtrTest, IsPtr_Range_ExceedsBuffer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    size_t too_large = str->capacity() + 10;
    
    EXPECT_FALSE(str->is_ptr(ptr, too_large));
}

TEST_F(StringIsPtrTest, IsPtr_Range_ExceedsFromMiddle) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 3;  // Start at 'l'
    
    EXPECT_FALSE(str->is_ptr(ptr, 10));  // 10 bytes would exceed buffer
}

TEST_F(StringIsPtrTest, IsPtr_Range_ExactlyAtEnd) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 3;
    size_t bytes = str->capacity() - 3;  // Exactly fits
    
    EXPECT_TRUE(str->is_ptr(ptr, bytes));
}

TEST_F(StringIsPtrTest, IsPtr_Range_OneByteOver) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 3;
    size_t bytes = str->capacity() - 3 + 1;  // One byte too many
    
    EXPECT_FALSE(str->is_ptr(ptr, bytes));
}

TEST_F(StringIsPtrTest, IsPtr_Range_NullPointer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_FALSE(str->is_ptr(nullptr, 5));
}

TEST_F(StringIsPtrTest, IsPtr_Range_ExternalPointer) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const char* external = "external";
    
    EXPECT_FALSE(str->is_ptr(external, 5));
}

// ================================================================================
// Empty String Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_EmptyString_BufferStart) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    
    EXPECT_TRUE(str->is_ptr(ptr));  // Still has buffer with null terminator
}

TEST_F(StringIsPtrTest, IsPtr_EmptyString_NullTerminator) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    // Even empty string has capacity for null terminator
    EXPECT_TRUE(str->is_ptr(str->c_str(), 1));  // Just the null terminator
}

TEST_F(StringIsPtrTest, IsPtr_EmptyString_ExceedsCapacity) {
    auto str = makeString("");
    ASSERT_NE(str.get(), nullptr);
    
    EXPECT_FALSE(str->is_ptr(str->c_str(), 100));  // Way more than capacity
}

// ================================================================================
// Capacity vs Length Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_WithinCapacity_BeyondLength) {
    auto str = makeString("hello", 100);  // Large capacity
    ASSERT_NE(str.get(), nullptr);
    
    // Pointer beyond string length but within capacity
    const void* ptr = str->c_str() + 50;
    
    EXPECT_TRUE(str->is_ptr(ptr));  // Within allocated buffer
}

TEST_F(StringIsPtrTest, IsPtr_Range_WithinCapacity_BeyondLength) {
    auto str = makeString("hello", 100);
    ASSERT_NE(str.get(), nullptr);
    
    // Range starts beyond length but within capacity
    const void* ptr = str->c_str() + 10;
    
    EXPECT_TRUE(str->is_ptr(ptr, 10));  // Still within buffer
}

TEST_F(StringIsPtrTest, IsPtr_BeyondCapacity) {
    auto str = makeString("hello", 100);
    ASSERT_NE(str.get(), nullptr);
    
    // Pointer beyond capacity
    const void* ptr = str->c_str() + 150;
    
    EXPECT_FALSE(str->is_ptr(ptr));
}

// ================================================================================
// Aliasing Detection Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_DetectSelfAliasing_Start) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    // Simulate concat with pointer to own buffer
    const char* self_ptr = str->c_str();
    
    if (str->is_ptr(self_ptr)) {
        EXPECT_TRUE(true);  // Correctly detected aliasing
    } else {
        FAIL() << "Should have detected aliasing";
    }
}

TEST_F(StringIsPtrTest, IsPtr_DetectSelfAliasing_Substring) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    // Pointer to substring within buffer
    const char* substring = str->c_str() + 6;  // Points to "world"
    
    EXPECT_TRUE(str->is_ptr(substring));
}

TEST_F(StringIsPtrTest, IsPtr_NoAliasing_ExternalString) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const char* external = "world";
    
    EXPECT_FALSE(str->is_ptr(external));  // Not aliased
}

// ================================================================================
// Multiple Pointer Checks
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_MultiplePointers_SomeValid_SomeInvalid) {
    auto str = makeString("hello world");
    ASSERT_NE(str.get(), nullptr);
    
    const void* p1 = str->c_str();           // Valid - start
    const void* p2 = str->c_str() + 5;       // Valid - middle
    const void* p3 = str->c_str() + 11;      // Valid - null terminator
    const char* p4 = "external";             // Invalid - external
    const void* p5 = str->c_str() + 100;     // Invalid - beyond buffer
    
    EXPECT_TRUE(str->is_ptr(p1));
    EXPECT_TRUE(str->is_ptr(p2));
    EXPECT_TRUE(str->is_ptr(p3));
    EXPECT_FALSE(str->is_ptr(p4));
    EXPECT_FALSE(str->is_ptr(p5));
}

// ================================================================================
// Long String Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_LongString_StartMiddleEnd) {
    std::string long_str(1000, 'x');
    auto str = makeString(long_str.c_str());
    ASSERT_NE(str.get(), nullptr);
    
    const void* start = str->c_str();
    const void* middle = str->c_str() + 500;
    const void* end = str->c_str() + 999;
    
    EXPECT_TRUE(str->is_ptr(start));
    EXPECT_TRUE(str->is_ptr(middle));
    EXPECT_TRUE(str->is_ptr(end));
}

TEST_F(StringIsPtrTest, IsPtr_LongString_LargeRange) {
    std::string long_str(1000, 'x');
    auto str = makeString(long_str.c_str());
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 100;
    
    EXPECT_TRUE(str->is_ptr(ptr, 500));   // 500 bytes from position 100
    EXPECT_FALSE(str->is_ptr(ptr, 2000)); // Would exceed buffer
}

// ================================================================================
// Overflow Safety Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_Range_OverflowSafe_MaxSize) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str() + 3;
    size_t huge_size = SIZE_MAX;
    
    // Should safely detect overflow without crashing
    EXPECT_FALSE(str->is_ptr(ptr, huge_size));
}

TEST_F(StringIsPtrTest, IsPtr_Range_OverflowSafe_NearMaxSize) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    size_t huge_size = SIZE_MAX - 1000;
    
    // Should safely detect overflow
    EXPECT_FALSE(str->is_ptr(ptr, huge_size));
}

// ================================================================================
// Integration with Other Methods Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_AfterConcat_OriginalPointerValid) {
    auto str = makeString("hello", 100);
    ASSERT_NE(str.get(), nullptr);
    
    const void* original_ptr = str->c_str();
    
    str->concat(" world");
    
    // Original pointer should still be valid (no reallocation)
    EXPECT_TRUE(str->is_ptr(original_ptr));
}

TEST_F(StringIsPtrTest, IsPtr_AfterReset_PointerStillValid) {
    auto str = makeString("hello", 100);
    ASSERT_NE(str.get(), nullptr);
    
    const void* ptr = str->c_str();
    
    str->reset();
    
    // Pointer to buffer should still be valid after reset
    EXPECT_TRUE(str->is_ptr(ptr));
}

TEST_F(StringIsPtrTest, IsPtr_AfterCopy_DifferentBuffers) {
    auto original = makeString("hello");
    ASSERT_NE(original.get(), nullptr);
    
    auto copy_r = original->copy();
    ASSERT_TRUE(copy_r.hasValue());
    
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
    
    // Pointers from original should NOT be in copy's buffer
    EXPECT_FALSE(copy->is_ptr(original->c_str()));
    EXPECT_FALSE(original->is_ptr(copy->c_str()));
}

// ================================================================================
// Pointer Arithmetic Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_PointerArithmetic_AllPositions) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Test every position in the buffer
    for (size_t i = 0; i < str->capacity(); ++i) {
        const void* ptr = str->c_str() + i;
        EXPECT_TRUE(str->is_ptr(ptr)) << "Position " << i << " should be valid";
    }
}

TEST_F(StringIsPtrTest, IsPtr_Range_AllValidRanges) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    // Test various valid ranges
    for (size_t start = 0; start < str->capacity(); ++start) {
        for (size_t len = 1; len <= str->capacity() - start; ++len) {
            const void* ptr = str->c_str() + start;
            EXPECT_TRUE(str->is_ptr(ptr, len)) 
                << "Range [" << start << ", " << start + len << ") should be valid";
        }
    }
}

// ================================================================================
// Edge Case: Two Different Strings
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_DifferentStrings_NotAliased) {
    auto str1 = makeString("hello");
    auto str2 = makeString("world");
    ASSERT_NE(str1.get(), nullptr);
    ASSERT_NE(str2.get(), nullptr);
    
    // Pointers from str1 should not be in str2's buffer
    EXPECT_FALSE(str2->is_ptr(str1->c_str()));
    EXPECT_FALSE(str1->is_ptr(str2->c_str()));
}

// ================================================================================
// Const Correctness Tests
// ================================================================================

TEST_F(StringIsPtrTest, IsPtr_ConstString_CanCallMethod) {
    auto str = makeString("hello");
    ASSERT_NE(str.get(), nullptr);
    
    const cslt::String* const_str = str.get();
    
    // Should be able to call is_ptr on const String
    EXPECT_TRUE(const_str->is_ptr(const_str->c_str()));
}
// -------------------------------------------------------------------------------- 

class StringFindTest : public ::testing::Test {
protected:
    cslt::HeapAllocator allocator;
    
    void SetUp() override {
        // Any setup needed before each test
    }
    
    void TearDown() override {
        // Any cleanup needed after each test
    }
    
    // Helper to create a string for testing
    cslt::UniquePtr<cslt::String, cslt::StringDeleter> 
    makeString(const char* str, size_t capacity = 0) {
        auto result = cslt::String::init(str, capacity, allocator);
        if (!result.hasValue()) {
            return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(nullptr);
        }
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(result.value());
    }
};

// ================================================================================
// Basic Find Tests - String Needle
// ================================================================================

TEST_F(StringFindTest, Find_String_Found_AtStart) {
    auto haystack = makeString("hello world");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);
}

TEST_F(StringFindTest, Find_String_Found_InMiddle) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 6u);
}

TEST_F(StringFindTest, Find_String_Found_AtEnd) {
    auto haystack = makeString("hello world");
    auto needle = makeString("d");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 10u);
}

TEST_F(StringFindTest, Find_String_NotFound) {
    auto haystack = makeString("hello world");
    auto needle = makeString("xyz");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);
}

TEST_F(StringFindTest, Find_String_EmptyNeedle) {
    auto haystack = makeString("hello world");
    auto needle = makeString("");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);  // Empty needle found at start
}

TEST_F(StringFindTest, Find_String_EmptyHaystack) {
    auto haystack = makeString("");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);  // Can't find in empty string
}

TEST_F(StringFindTest, Find_String_BothEmpty) {
    auto haystack = makeString("");
    auto needle = makeString("");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);  // Empty needle found at start
}

TEST_F(StringFindTest, Find_String_NeedleLongerThanHaystack) {
    auto haystack = makeString("hi");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);
}

TEST_F(StringFindTest, Find_String_MultipleOccurrences_ReturnsFirst) {
    auto haystack = makeString("hello hello hello");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);  // Returns first occurrence
}

// ================================================================================
// Basic Find Tests - C-String Needle
// ================================================================================

TEST_F(StringFindTest, Find_CString_Found_AtStart) {
    auto haystack = makeString("hello world");
    ASSERT_NE(haystack.get(), nullptr);
    
    size_t pos = haystack->find("hello");
    EXPECT_EQ(pos, 0u);
}

TEST_F(StringFindTest, Find_CString_Found_InMiddle) {
    auto haystack = makeString("hello world");
    ASSERT_NE(haystack.get(), nullptr);
    
    size_t pos = haystack->find("world");
    EXPECT_EQ(pos, 6u);
}

TEST_F(StringFindTest, Find_CString_NotFound) {
    auto haystack = makeString("hello world");
    ASSERT_NE(haystack.get(), nullptr);
    
    size_t pos = haystack->find("xyz");
    EXPECT_EQ(pos, SIZE_MAX);
}

TEST_F(StringFindTest, Find_CString_EmptyNeedle) {
    auto haystack = makeString("hello world");
    ASSERT_NE(haystack.get(), nullptr);
    
    size_t pos = haystack->find("");
    EXPECT_EQ(pos, 0u);
}

TEST_F(StringFindTest, Find_CString_SingleCharacter) {
    auto haystack = makeString("hello world");
    ASSERT_NE(haystack.get(), nullptr);
    
    size_t pos = haystack->find("o");
    EXPECT_EQ(pos, 4u);  // First 'o' in "hello"
}

// ================================================================================
// Forward vs Reverse Search Tests
// ================================================================================

TEST_F(StringFindTest, Find_Forward_MultipleOccurrences) {
    auto haystack = makeString("hello world hello");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle, nullptr, nullptr, FORWARD);
    EXPECT_EQ(pos, 0u);  // First occurrence
}

TEST_F(StringFindTest, Find_Reverse_MultipleOccurrences) {
    auto haystack = makeString("hello world hello");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle, nullptr, nullptr, REVERSE);
    EXPECT_EQ(pos, 12u);  // Last occurrence
}

TEST_F(StringFindTest, Find_Reverse_SingleOccurrence) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle, nullptr, nullptr, REVERSE);
    EXPECT_EQ(pos, 6u);  // Same as forward for single occurrence
}

TEST_F(StringFindTest, Find_Reverse_NotFound) {
    auto haystack = makeString("hello world");
    auto needle = makeString("xyz");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle, nullptr, nullptr, REVERSE);
    EXPECT_EQ(pos, SIZE_MAX);
}

// ================================================================================
// Range-Based Search Tests
// ================================================================================

TEST_F(StringFindTest, Find_WithBegin_FindsAfterStart) {
    auto haystack = makeString("hello world hello");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* begin = haystack->c_str() + 1;
    size_t pos = haystack->find(*needle, begin);
    EXPECT_EQ(pos, 12u);  // Skips first "hello", finds second
}

TEST_F(StringFindTest, Find_WithBeginAndEnd_InRange) {
    auto haystack = makeString("hello world hello");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* begin = haystack->c_str() + 5;
    const void* end = haystack->c_str() + 12;
    size_t pos = haystack->find(*needle, begin, end);
    std::cout << pos << "\n";
    EXPECT_EQ(pos, 6u);  // Found within range
}

TEST_F(StringFindTest, Find_WithBeginAndEnd_OutsideRange) {
    auto haystack = makeString("hello world hello");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* begin = haystack->c_str() + 1;
    const void* end = haystack->c_str() + 10;
    size_t pos = haystack->find(*needle, begin, end);
    EXPECT_EQ(pos, SIZE_MAX);  // Second "hello" is outside range
}

TEST_F(StringFindTest, Find_WithEnd_TruncatesSearch) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* end = haystack->c_str() + 5;
    size_t pos = haystack->find(*needle, nullptr, end);
    EXPECT_EQ(pos, SIZE_MAX);  // "world" is beyond end
}

TEST_F(StringFindTest, Find_RangeExactlyContainsNeedle) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* begin = haystack->c_str() + 6;
    const void* end = haystack->c_str() + 11;
    size_t pos = haystack->find(*needle, begin, end);
    EXPECT_EQ(pos, 6u);
}

// ================================================================================
// Invalid Range Tests
// ================================================================================

TEST_F(StringFindTest, Find_InvalidRange_BeginAfterEnd) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const void* begin = haystack->c_str() + 10;
    const void* end = haystack->c_str() + 5;
    size_t pos = haystack->find(*needle, begin, end);
    EXPECT_EQ(pos, SIZE_MAX);  // Invalid range
}

TEST_F(StringFindTest, Find_InvalidRange_BeginOutsideBuffer) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const char* external = "external";
    size_t pos = haystack->find(*needle, external);
    EXPECT_EQ(pos, SIZE_MAX);  // Invalid pointer
}

TEST_F(StringFindTest, Find_InvalidRange_EndOutsideBuffer) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    const char* external = "external";
    size_t pos = haystack->find(*needle, nullptr, external);
    EXPECT_EQ(pos, SIZE_MAX);  // Invalid pointer
}

// ================================================================================
// Case Sensitivity Tests
// ================================================================================

TEST_F(StringFindTest, Find_CaseSensitive) {
    auto haystack = makeString("Hello World");
    auto needle = makeString("hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);  // Case sensitive - won't find
}

TEST_F(StringFindTest, Find_CaseSensitive_ExactMatch) {
    auto haystack = makeString("Hello World");
    auto needle = makeString("Hello");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);  // Exact case match
}

// ================================================================================
// Special Characters Tests
// ================================================================================

TEST_F(StringFindTest, Find_WithNewline) {
    auto haystack = makeString("hello\nworld");
    auto needle = makeString("\n");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 5u);
}

TEST_F(StringFindTest, Find_WithTab) {
    auto haystack = makeString("hello\tworld");
    auto needle = makeString("\t");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 5u);
}

TEST_F(StringFindTest, Find_WithNullInMiddle) {
    // String with embedded null
    char buffer[] = {'h', 'e', 'l', 'l', 'o', '\0', 'w', 'o', 'r', 'l', 'd', '\0'};
    auto haystack = makeString(buffer);  // Will stop at first null
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);  // String is only "hello" due to null terminator
}

// ================================================================================
// Long String Tests
// ================================================================================

TEST_F(StringFindTest, Find_LongString_Found) {
    std::string long_str(1000, 'x');
    long_str += "needle";
    long_str += std::string(1000, 'y');
    
    auto haystack = makeString(long_str.c_str());
    auto needle = makeString("needle");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 1000u);
}

TEST_F(StringFindTest, Find_LongString_NotFound) {
    std::string long_str(10000, 'x');
    
    auto haystack = makeString(long_str.c_str());
    auto needle = makeString("needle");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, SIZE_MAX);
}

// ================================================================================
// Overlapping Pattern Tests
// ================================================================================

TEST_F(StringFindTest, Find_OverlappingPattern) {
    auto haystack = makeString("aaaa");
    auto needle = makeString("aaa");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);  // Returns first match
}

TEST_F(StringFindTest, Find_RepeatingPattern) {
    auto haystack = makeString("ababababab");
    auto needle = makeString("abab");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos = haystack->find(*needle);
    EXPECT_EQ(pos, 0u);
}

// ================================================================================
// Integration Tests
// ================================================================================

TEST_F(StringFindTest, Find_AfterConcat) {
    auto haystack = makeString("hello", 100);
    ASSERT_NE(haystack.get(), nullptr);
    
    haystack->concat(" world");
    
    size_t pos = haystack->find("world");
    EXPECT_EQ(pos, 6u);
}

TEST_F(StringFindTest, Find_AfterReset) {
    auto haystack = makeString("hello world", 100);
    ASSERT_NE(haystack.get(), nullptr);
    
    haystack->reset();
    haystack->concat("new content");
    
    size_t pos = haystack->find("new");
    EXPECT_EQ(pos, 0u);
}

// ================================================================================
// Comparison: String vs C-String Needle
// ================================================================================

TEST_F(StringFindTest, Find_StringAndCString_SameResult) {
    auto haystack = makeString("hello world");
    auto needle = makeString("world");
    ASSERT_NE(haystack.get(), nullptr);
    ASSERT_NE(needle.get(), nullptr);
    
    size_t pos1 = haystack->find(*needle);
    size_t pos2 = haystack->find("world");
    
    EXPECT_EQ(pos1, pos2);
}
// ================================================================================ 
// ================================================================================ 

/**
 * @brief Test fixture that provides a shared HeapAllocator and a helper for
 *        constructing String instances without boilerplate in every test.
 */
class StringWordsTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;

    /**
     * @brief Construct a String from a C-string literal, asserting on failure.
     *        Returns an owning UniquePtr so the caller's local variable manages
     *        lifetime automatically.
     */
    cslt::UniquePtr<cslt::String, cslt::StringDeleter>
    make(const char* cstr) {
        auto r = cslt::String::init(cstr, 0, alloc);
        EXPECT_TRUE(r.hasValue()) << "String::init failed for: " << cstr;
        return cslt::UniquePtr<cslt::String, cslt::StringDeleter>(r.value());
    }
};

// ================================================================================
// words(const String&) — String overload
// ================================================================================

// Single occurrence — word appears exactly once
TEST_F(StringWordsTest, StringOverload_SingleOccurrence) {
    auto haystack = make("the quick brown fox");
    auto needle   = make("fox");
    EXPECT_EQ(haystack->words(*needle), 1u);
}

// Multiple non-overlapping occurrences
TEST_F(StringWordsTest, StringOverload_MultipleOccurrences) {
    auto haystack = make("one fish two fish red fish blue fish");
    auto needle   = make("fish");
    EXPECT_EQ(haystack->words(*needle), 4u);
}

// Word not present — should return 0
TEST_F(StringWordsTest, StringOverload_NotFound) {
    auto haystack = make("hello world");
    auto needle   = make("cat");
    EXPECT_EQ(haystack->words(*needle), 0u);
}

// Search is case-sensitive — uppercase variant must not match lowercase
TEST_F(StringWordsTest, StringOverload_CaseSensitive) {
    auto haystack = make("Hello hello HELLO");
    auto needle   = make("hello");
    EXPECT_EQ(haystack->words(*needle), 1u);
}

// Word at the very start of the string
TEST_F(StringWordsTest, StringOverload_MatchAtStart) {
    auto haystack = make("fish and chips");
    auto needle   = make("fish");
    EXPECT_EQ(haystack->words(*needle), 1u);
}

// Word at the very end of the string
TEST_F(StringWordsTest, StringOverload_MatchAtEnd) {
    auto haystack = make("salt and fish");
    auto needle   = make("fish");
    EXPECT_EQ(haystack->words(*needle), 1u);
}

// Haystack and needle are identical
TEST_F(StringWordsTest, StringOverload_HaystackEqualsNeedle) {
    auto haystack = make("fish");
    auto needle   = make("fish");
    EXPECT_EQ(haystack->words(*needle), 1u);
}

// Needle longer than haystack — impossible to match
TEST_F(StringWordsTest, StringOverload_NeedleLongerThanHaystack) {
    auto haystack = make("hi");
    auto needle   = make("hello world");
    EXPECT_EQ(haystack->words(*needle), 0u);
}

// Restrict search to a sub-range that contains exactly one of two occurrences
TEST_F(StringWordsTest, StringOverload_WithRangeOneOfTwo) {
    auto haystack = make("fish chips fish");
    auto needle   = make("fish");

    // Range covers only the first five characters ("fish ")
    const void* begin = haystack->c_str();
    const void* end   = haystack->c_str() + 5;

    EXPECT_EQ(haystack->words(*needle, begin, end), 1u);
}

// Restrict search to a sub-range that contains no occurrences
TEST_F(StringWordsTest, StringOverload_WithRangeNoMatch) {
    auto haystack = make("fish chips fish");
    auto needle   = make("fish");

    // Middle section "chips" contains no "fish"
    const void* begin = haystack->c_str() + 5;
    const void* end   = haystack->c_str() + 10;

    EXPECT_EQ(haystack->words(*needle, begin, end), 0u);
}

// ================================================================================
// words(const char*) — C-string literal overload
// ================================================================================

// Single occurrence
TEST_F(StringWordsTest, LiteralOverload_SingleOccurrence) {
    auto haystack = make("the quick brown fox");
    EXPECT_EQ(haystack->words("fox"), 1u);
}

// Multiple non-overlapping occurrences
TEST_F(StringWordsTest, LiteralOverload_MultipleOccurrences) {
    auto haystack = make("one fish two fish red fish blue fish");
    EXPECT_EQ(haystack->words("fish"), 4u);
}

// Word not present — should return 0
TEST_F(StringWordsTest, LiteralOverload_NotFound) {
    auto haystack = make("hello world");
    EXPECT_EQ(haystack->words("cat"), 0u);
}

// Search is case-sensitive
TEST_F(StringWordsTest, LiteralOverload_CaseSensitive) {
    auto haystack = make("Hello hello HELLO");
    EXPECT_EQ(haystack->words("hello"), 1u);
}

// Word at the very start of the string
TEST_F(StringWordsTest, LiteralOverload_MatchAtStart) {
    auto haystack = make("fish and chips");
    EXPECT_EQ(haystack->words("fish"), 1u);
}

// Word at the very end of the string
TEST_F(StringWordsTest, LiteralOverload_MatchAtEnd) {
    auto haystack = make("salt and fish");
    EXPECT_EQ(haystack->words("fish"), 1u);
}

// Haystack and needle are identical
TEST_F(StringWordsTest, LiteralOverload_HaystackEqualsNeedle) {
    auto haystack = make("fish");
    EXPECT_EQ(haystack->words("fish"), 1u);
}

// Needle longer than haystack — impossible to match
TEST_F(StringWordsTest, LiteralOverload_NeedleLongerThanHaystack) {
    auto haystack = make("hi");
    EXPECT_EQ(haystack->words("hello world"), 0u);
}

// Restrict search to a sub-range that contains exactly one of two occurrences
TEST_F(StringWordsTest, LiteralOverload_WithRangeOneOfTwo) {
    auto haystack = make("fish chips fish");

    // Range covers only the first five characters ("fish ")
    const void* begin = haystack->c_str();
    const void* end   = haystack->c_str() + 5;

    EXPECT_EQ(haystack->words("fish", begin, end), 1u);
}

// Restrict search to a sub-range that contains no occurrences
TEST_F(StringWordsTest, LiteralOverload_WithRangeNoMatch) {
    auto haystack = make("fish chips fish");

    // Middle section "chips" contains no "fish"
    const void* begin = haystack->c_str() + 5;
    const void* end   = haystack->c_str() + 10;

    EXPECT_EQ(haystack->words("fish", begin, end), 0u);
}

// ================================================================================
// Cross-overload consistency
// ================================================================================

// Both overloads must agree on the same haystack
TEST_F(StringWordsTest, BothOverloads_AgreeOnCount) {
    auto haystack = make("to be or not to be that is the question");
    auto needle   = make("be");

    size_t from_string  = haystack->words(*needle);
    size_t from_literal = haystack->words("be");

    EXPECT_EQ(from_string, from_literal);
    EXPECT_EQ(from_string, 2u);
}
// ================================================================================
// ================================================================================
// eof
