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

#include <gtest/gtest.h>
#include <cstring>

using namespace cslt;
// ================================================================================ 
// ================================================================================ 

// Test fixture for HeapAllocator
class HeapAllocatorTest : public ::testing::Test {
protected:
    HeapAllocator allocator;
    
    // Helper function to check if memory is zeroed
    bool is_zeroed(void* ptr, size_t bytes) {
        unsigned char* p = static_cast<unsigned char*>(ptr);
        for (size_t i = 0; i < bytes; ++i) {
            if (p[i] != 0) return false;
        }
        return true;
    }
    
    // Helper function to check if pointer is aligned
    bool is_aligned(void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
};
// ================================================================================ 
// ================================================================================

TEST_F(HeapAllocatorTest, BasicAllocation) {
    auto result = allocator.alloc(1024, false);
    ASSERT_TRUE(result.hasValue());
    ASSERT_NE(result.value(), nullptr);
    
    // Write some data
    memset(result.value(), 0xAB, 1024);
    
    // Clean up
    allocator.return_element(result.value(), 1024, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, AllocZeroed) {
    auto result = allocator.alloc(512, true);
    ASSERT_TRUE(result.hasValue());
    ASSERT_NE(result.value(), nullptr);
    EXPECT_TRUE(is_zeroed(result.value(), 512));
    
    allocator.return_element(result.value(), 512, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, AllocZeroBytes) {
    auto result = allocator.alloc(0, false);
    EXPECT_FALSE(result.hasValue());
}
// ================================================================================
// ================================================================================

TEST_F(HeapAllocatorTest, AlignedAllocation) {
    size_t alignments[] = {16, 32, 64, 128, 256};
    
    for (size_t align : alignments) {
        auto result = allocator.alloc_aligned(1024, align, false);
        ASSERT_TRUE(result.hasValue());
        ASSERT_NE(result.value(), nullptr);
        EXPECT_TRUE(is_aligned(result.value(), align)) 
            << "Allocation not aligned to " << align << " bytes";
        
        allocator.return_element(result.value(), 1024, align);
    }
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, AlignedAllocationZeroed) {
    auto result = allocator.alloc_aligned(1024, 64, true);
    ASSERT_TRUE(result.hasValue());
    ASSERT_NE(result.value(), nullptr);
    EXPECT_TRUE(is_aligned(result.value(), 64));
    EXPECT_TRUE(is_zeroed(result.value(), 1024));
    
    allocator.return_element(result.value(), 1024, 64);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, AlignedAllocationDefaultAlignment) {
    auto result = allocator.alloc_aligned(1024, 0, false);  // 0 should use default
    ASSERT_TRUE(result.hasValue());
    ASSERT_NE(result.value(), nullptr);
    
    allocator.return_element(result.value(), 1024, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, AlignedAllocationZeroBytes) {
    auto result = allocator.alloc_aligned(0, 64, false);
    EXPECT_FALSE(result.hasValue());
}
// ================================================================================
// ================================================================================

TEST_F(HeapAllocatorTest, ReallocGrow) {
    // Allocate initial block
    auto result1 = allocator.alloc(512, false);
    ASSERT_TRUE(result1.hasValue());
    
    // Fill with pattern
    unsigned char* p = static_cast<unsigned char*>(result1.value());
    for (size_t i = 0; i < 512; ++i) {
        p[i] = static_cast<unsigned char>(i % 256);
    }
    
    // Reallocate to larger size
    auto result2 = allocator.realloc(result1.value(), 512, 1024, false);
    ASSERT_TRUE(result2.hasValue());
    ASSERT_NE(result2.value(), nullptr);
    
    // Verify data was copied
    unsigned char* p2 = static_cast<unsigned char*>(result2.value());
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(p2[i], static_cast<unsigned char>(i % 256)) 
            << "Data mismatch at byte " << i;
    }
    
    allocator.return_element(result2.value(), 1024, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocShrink) {
    // Allocate initial block
    auto result1 = allocator.alloc(1024, false);
    ASSERT_TRUE(result1.hasValue());
    
    // Fill with pattern
    unsigned char* p = static_cast<unsigned char*>(result1.value());
    for (size_t i = 0; i < 1024; ++i) {
        p[i] = static_cast<unsigned char>(i % 256);
    }
    
    // Reallocate to smaller size
    auto result2 = allocator.realloc(result1.value(), 1024, 512, false);
    ASSERT_TRUE(result2.hasValue());
    ASSERT_NE(result2.value(), nullptr);
    
    // Verify data was copied (first 512 bytes)
    unsigned char* p2 = static_cast<unsigned char*>(result2.value());
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(p2[i], static_cast<unsigned char>(i % 256))
            << "Data mismatch at byte " << i;
    }
    
    allocator.return_element(result2.value(), 512, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocZeroed) {
    // Allocate initial block
    auto result1 = allocator.alloc(512, false);
    ASSERT_TRUE(result1.hasValue());
    
    // Fill with pattern
    memset(result1.value(), 0xFF, 512);
    
    // Reallocate to larger size with zeroing
    auto result2 = allocator.realloc(result1.value(), 512, 1024, true);
    ASSERT_TRUE(result2.hasValue());
    
    unsigned char* p = static_cast<unsigned char*>(result2.value());
    
    // Verify old data is preserved
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(p[i], 0xFF) << "Old data not preserved at byte " << i;
    }
    
    // Verify new bytes are zeroed
    for (size_t i = 512; i < 1024; ++i) {
        EXPECT_EQ(p[i], 0) << "New bytes not zeroed at byte " << i;
    }
    
    allocator.return_element(result2.value(), 1024, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocNullPointer) {
    auto result = allocator.realloc(nullptr, 0, 1024, false);
    EXPECT_FALSE(result.hasValue());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocZeroBytes) {
    auto result1 = allocator.alloc(512, false);
    ASSERT_TRUE(result1.hasValue());
    
    auto result2 = allocator.realloc(result1.value(), 512, 0, false);
    EXPECT_FALSE(result2.hasValue());
    
    // Clean up original allocation
    allocator.return_element(result1.value(), 512, allocator.default_alignment());
}
// ================================================================================
// ================================================================================

TEST_F(HeapAllocatorTest, ReallocAligned) {
    // Allocate initial aligned block
    auto result1 = allocator.alloc_aligned(512, 64, false);
    ASSERT_TRUE(result1.hasValue());
    
    // Fill with pattern
    unsigned char* p = static_cast<unsigned char*>(result1.value());
    for (size_t i = 0; i < 512; ++i) {
        p[i] = static_cast<unsigned char>(i % 256);
    }
    
    // Reallocate to larger size
    auto result2 = allocator.realloc_aligned(result1.value(), 512, 1024, 64, false);
    ASSERT_TRUE(result2.hasValue());
    EXPECT_TRUE(is_aligned(result2.value(), 64));
    
    // Verify data was copied
    unsigned char* p2 = static_cast<unsigned char*>(result2.value());
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(p2[i], static_cast<unsigned char>(i % 256))
            << "Data mismatch at byte " << i;
    }
    
    allocator.return_element(result2.value(), 1024, 64);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocAlignedZeroed) {
    // Allocate initial aligned block
    auto result1 = allocator.alloc_aligned(512, 128, false);
    ASSERT_TRUE(result1.hasValue());
    memset(result1.value(), 0xAA, 512);
    
    // Reallocate to larger size with zeroing
    auto result2 = allocator.realloc_aligned(result1.value(), 512, 1024, 128, true);
    ASSERT_TRUE(result2.hasValue());
    EXPECT_TRUE(is_aligned(result2.value(), 128));
    
    unsigned char* p = static_cast<unsigned char*>(result2.value());
    
    // Verify old data preserved
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(p[i], 0xAA) << "Old data not preserved at byte " << i;
    }
    
    // Verify new bytes zeroed
    for (size_t i = 512; i < 1024; ++i) {
        EXPECT_EQ(p[i], 0) << "New bytes not zeroed at byte " << i;
    }
    
    allocator.return_element(result2.value(), 1024, 128);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocAlignedNullPointer) {
    auto result = allocator.realloc_aligned(nullptr, 0, 1024, 64, false);
    EXPECT_FALSE(result.hasValue());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ReallocAlignedZeroBytes) {
    auto result1 = allocator.alloc_aligned(512, 64, false);
    ASSERT_TRUE(result1.hasValue());
    
    auto result2 = allocator.realloc_aligned(result1.value(), 512, 0, 64, false);
    EXPECT_FALSE(result2.hasValue());
    
    // Clean up original allocation
    allocator.return_element(result1.value(), 512, 64);
}
// ================================================================================
// ================================================================================

TEST_F(HeapAllocatorTest, MultipleSimultaneousAllocations) {
    const size_t num_allocs = 10;
    Expected<void*> results[num_allocs];
    
    // Allocate multiple blocks
    for (size_t i = 0; i < num_allocs; ++i) {
        results[i] = allocator.alloc(256, false);
        ASSERT_TRUE(results[i].hasValue());
        
        // Fill each with unique pattern
        memset(results[i].value(), static_cast<int>(i), 256);
    }
    
    // Verify each block still has its pattern
    for (size_t i = 0; i < num_allocs; ++i) {
        unsigned char* p = static_cast<unsigned char*>(results[i].value());
        for (size_t j = 0; j < 256; ++j) {
            EXPECT_EQ(p[j], static_cast<unsigned char>(i))
                << "Data corruption in allocation " << i << " at byte " << j;
        }
    }
    
    // Clean up
    for (size_t i = 0; i < num_allocs; ++i) {
        allocator.return_element(results[i].value(), 256, allocator.default_alignment());
    }
}
// ================================================================================
// ================================================================================

TEST_F(HeapAllocatorTest, LargeAllocation) {
    const size_t large_size = 10 * 1024 * 1024;  // 10 MB
    auto result = allocator.alloc(large_size, false);
    ASSERT_TRUE(result.hasValue());
    ASSERT_NE(result.value(), nullptr);
    
    allocator.return_element(result.value(), large_size, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorTest, ManySmallAllocations) {
    const size_t num_allocs = 1000;
    std::vector<Expected<void*>> results;
    
    for (size_t i = 0; i < num_allocs; ++i) {
        auto result = allocator.alloc(32, false);
        ASSERT_TRUE(result.hasValue());
        results.push_back(result);
    }
    
    // Clean up
    for (auto& result : results) {
        allocator.return_element(result.value(), 32, allocator.default_alignment());
    }
}
// ================================================================================ 
// ================================================================================ 

class HeapAllocatorExtendedTest : public ::testing::Test {
protected:
    HeapAllocator allocator;
};
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, StatsBasic) {
    char buffer[512];
    
    bool result = allocator.stats(buffer, sizeof(buffer));
    ASSERT_TRUE(result);
    
    // Check that buffer contains expected strings
    EXPECT_NE(strstr(buffer, "HeapAllocator Statistics"), nullptr);
    EXPECT_NE(strstr(buffer, "Type: DYNAMIC"), nullptr);
    EXPECT_NE(strstr(buffer, "Default Alignment:"), nullptr);
    EXPECT_NE(strstr(buffer, "System Heap"), nullptr);
    EXPECT_NE(strstr(buffer, "wrapper"), nullptr);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, StatsNullBuffer) {
    bool result = allocator.stats(nullptr, 512);
    EXPECT_FALSE(result);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, StatsZeroSize) {
    char buffer[512];
    bool result = allocator.stats(buffer, 0);
    EXPECT_FALSE(result);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, StatsBufferTooSmall) {
    char buffer[10];  // Way too small
    bool result = allocator.stats(buffer, sizeof(buffer));
    EXPECT_FALSE(result);  // Should fail due to insufficient space
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, StatsPrintOutput) {
    char buffer[512];
    
    if (allocator.stats(buffer, sizeof(buffer))) {
        // This will print during test execution with --verbose
        printf("\n%s\n", buffer);
    }
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, MemoryType) {
    MemType type = allocator.memory_type();
    EXPECT_EQ(type, DYNAMIC);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, OwnsMemory) {
    bool owns = allocator.owns_memory();
    EXPECT_FALSE(owns);  // HeapAllocator doesn't own memory
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, Size) {
    size_t size = allocator.size();
    EXPECT_EQ(size, 0);  // HeapAllocator doesn't track size
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, TotalAlloc) {
    size_t total = allocator.total_alloc();
    EXPECT_EQ(total, 0);  // HeapAllocator has no fixed allocation
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, Remaining) {
    size_t remaining = allocator.remaining();
    EXPECT_EQ(remaining, 0);  // HeapAllocator doesn't track remaining
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, IsPtrReturnsFalse) {
    auto result = allocator.alloc(128, false);
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    
    // HeapAllocator can't verify pointers
    EXPECT_FALSE(allocator.is_ptr(ptr));
    EXPECT_FALSE(allocator.is_ptr(nullptr));
    
    allocator.return_element(ptr, 128, allocator.default_alignment());
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, IsPtrSizedReturnsFalse) {
    auto result = allocator.alloc(128, false);
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    
    // HeapAllocator can't verify sized pointers
    EXPECT_FALSE(allocator.is_ptr_sized(ptr, 128));
    EXPECT_FALSE(allocator.is_ptr_sized(nullptr, 128));
    
    allocator.return_element(ptr, 128, allocator.default_alignment());
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, SaveReturnsNull) {
    void* checkpoint = allocator.save();
    EXPECT_EQ(checkpoint, nullptr);  // HeapAllocator doesn't support save
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, RestoreReturnsFalse) {
    void* fake_checkpoint = reinterpret_cast<void*>(0x1234);
    bool result = allocator.restore(fake_checkpoint);
    EXPECT_FALSE(result);  // HeapAllocator doesn't support restore
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, ResetDoesNothing) {
    // Allocate some memory
    auto result = allocator.alloc(128, false);
    ASSERT_TRUE(result.hasValue());
    void* ptr = result.value();
    
    // Reset should be a no-op
    allocator.reset();
    
    // Memory should still be valid (reset doesn't free it)
    memset(ptr, 0xFF, 128);  // Should not crash
    
    // Clean up
    allocator.return_element(ptr, 128, allocator.default_alignment());
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, DefaultAlignment) {
    size_t alignment = allocator.default_alignment();
    EXPECT_EQ(alignment, alignof(max_align_t));
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, CustomAlignment) {
    HeapAllocator custom_allocator(64);
    EXPECT_EQ(custom_allocator.default_alignment(), 64);
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, FullWorkflow) {
    char stats_buffer[512];
    
    // Check initial state
    EXPECT_EQ(allocator.memory_type(), DYNAMIC);
    EXPECT_FALSE(allocator.owns_memory());
    EXPECT_EQ(allocator.size(), 0);
    EXPECT_EQ(allocator.total_alloc(), 0);
    EXPECT_EQ(allocator.remaining(), 0);
    
    // Allocate some memory
    auto result1 = allocator.alloc(256, true);
    ASSERT_TRUE(result1.hasValue());
    void* ptr1 = result1.value();
    
    // Allocate aligned memory
    auto result2 = allocator.alloc_aligned(512, 64, false);
    ASSERT_TRUE(result2.hasValue());
    void* ptr2 = result2.value();
    
    // Note: HeapAllocator doesn't track these allocations in size()
    EXPECT_EQ(allocator.size(), 0);
    
    // Verify pointers (should return false since HeapAllocator can't verify)
    EXPECT_FALSE(allocator.is_ptr(ptr1));
    EXPECT_FALSE(allocator.is_ptr_sized(ptr2, 512));
    
    // Get stats
    bool stats_result = allocator.stats(stats_buffer, sizeof(stats_buffer));
    ASSERT_TRUE(stats_result);
    
    // Print stats (optional, for manual verification)
    printf("\n=== HeapAllocator Stats ===\n%s\n", stats_buffer);
    
    // Save/restore not supported
    EXPECT_EQ(allocator.save(), nullptr);
    EXPECT_FALSE(allocator.restore(nullptr));
    
    // Reset is a no-op
    allocator.reset();
    
    // Clean up
    allocator.return_element(ptr1, 256, allocator.default_alignment());
    allocator.return_element(ptr2, 512, 64);
}
// ================================================================================ 
// ================================================================================ 

TEST_F(HeapAllocatorExtendedTest, StatsWithDefaultAlignment16) {
    HeapAllocator alloc16(16);
    char buffer[512];
    
    ASSERT_TRUE(alloc16.stats(buffer, sizeof(buffer)));
    EXPECT_NE(strstr(buffer, "16 bytes"), nullptr);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, StatsWithCustomAlignment) {
    HeapAllocator alloc128(128);
    char buffer[512];
    
    ASSERT_TRUE(alloc128.stats(buffer, sizeof(buffer)));
    EXPECT_NE(strstr(buffer, "128 bytes"), nullptr);
}
// -------------------------------------------------------------------------------- 

TEST_F(HeapAllocatorExtendedTest, RemainingWithZeroAlloc) {
    // When alloc_ is 0 (HeapAllocator default)
    size_t remaining = allocator.remaining();
    EXPECT_EQ(remaining, 0);
}
// ================================================================================
// ================================================================================
// eof
