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
// ================================================================================
// ================================================================================
// eof
// ================================================================================
// ================================================================================
// eof
// ================================================================================
// ================================================================================
// eof
// ================================================================================
// ================================================================================
// eof
