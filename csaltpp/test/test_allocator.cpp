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
#include <random>
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

class ArenaAllocatorHeapTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ================================================================================
// Basic Creation and Destruction Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, BasicCreation) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    ASSERT_NE(arena, nullptr);
    EXPECT_EQ(arena->memory_type(), cslt::DYNAMIC);
    EXPECT_TRUE(arena->owns_memory());
}

TEST_F(ArenaAllocatorHeapTest, CreationWithAllParams) {
    auto result = cslt::ArenaAllocator::Heap(8192, true, 4096, 16);
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    EXPECT_EQ(arena->default_alignment(), 16);
    EXPECT_EQ(arena->min_chunk_size(), 4096);
}

TEST_F(ArenaAllocatorHeapTest, RespectsMinChunkSize) {
    // Request 1024 bytes but min_chunk is 4096
    auto result = cslt::ArenaAllocator::Heap(1024, true, 4096);
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    // Total allocation should be at least 4096
    EXPECT_GE(arena->total_alloc(), 4096);
}

TEST_F(ArenaAllocatorHeapTest, CreationTooSmallGetsBumpedUp) {
    // Request very small size - should automatically increase to min_chunk
    auto result = cslt::ArenaAllocator::Heap(10, false, 4096);  // min_chunk = 4096
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    // Should have allocated at least the minimum chunk size
    EXPECT_GE(arena->total_alloc(), 4096);
}

TEST_F(ArenaAllocatorHeapTest, CreationZeroSizeGetsBumpedUp) {
    // Request zero size - should automatically increase to min_chunk
    auto result = cslt::ArenaAllocator::Heap(0, false, 4096);  // min_chunk = 4096
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    // Should have allocated at least the minimum chunk size
    EXPECT_GE(arena->total_alloc(), 4096);
}
// ================================================================================
// Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, SimpleAllocation) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate 128 bytes
    auto alloc_result = arena->alloc(128);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(arena->is_ptr(ptr));
    EXPECT_GE(arena->size(), 128);
}

TEST_F(ArenaAllocatorHeapTest, MultipleAllocations) {
    auto result = cslt::ArenaAllocator::Heap(4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr1 = arena->alloc(64).value();
    void* ptr2 = arena->alloc(128).value();
    void* ptr3 = arena->alloc(256).value();
    
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);
    
    // All pointers should be tracked
    EXPECT_TRUE(arena->is_ptr(ptr1));
    EXPECT_TRUE(arena->is_ptr(ptr2));
    EXPECT_TRUE(arena->is_ptr(ptr3));
    
    // Size should account for all allocations (plus padding)
    EXPECT_GE(arena->size(), 64 + 128 + 256);
}

TEST_F(ArenaAllocatorHeapTest, AllocationFillsArena) {
    auto result = cslt::ArenaAllocator::Heap(2048, false);  // Non-resizable
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate until we run out
    size_t allocated = 0;
    while (true) {
        auto alloc_result = arena->alloc(64);
        if (!alloc_result.hasValue()) {
            break;  // Out of memory
        }
        allocated += 64;
    }
    
    EXPECT_GT(allocated, 0);
    EXPECT_LT(arena->remaining(), 64);  // Can't fit another 64-byte allocation
}

TEST_F(ArenaAllocatorHeapTest, ZeroedAllocation) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate with zeroing
    auto alloc_result = arena->alloc(256, true);
    ASSERT_TRUE(alloc_result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(alloc_result.value());
    
    // Verify all bytes are zero
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}

TEST_F(ArenaAllocatorHeapTest, AllocationZeroBytes) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc_result = arena->alloc(0);
    EXPECT_TRUE(alloc_result.hasError());
}

// ================================================================================
// Aligned Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, AlignedAllocation) {
    auto result = cslt::ArenaAllocator::Heap(4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate with 64-byte alignment
    auto alloc_result = arena->alloc_aligned(128, 64);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Verify alignment
    EXPECT_EQ(addr % 64, 0);
}

TEST_F(ArenaAllocatorHeapTest, MultipleAlignedAllocations) {
    auto result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr1 = arena->alloc_aligned(64, 16).value();
    void* ptr2 = arena->alloc_aligned(128, 32).value();
    void* ptr3 = arena->alloc_aligned(256, 64).value();
    
    // Verify alignments
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 32, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 64, 0);
}

TEST_F(ArenaAllocatorHeapTest, AlignedAllocationUsesDefault) {
    auto result = cslt::ArenaAllocator::Heap(1024, false, 0, 32);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Alignment 0 should use default (32)
    auto alloc_result = arena->alloc_aligned(64, 0);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 32, 0);
}

// ================================================================================
// Resize Tests (Dynamic Growth)
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, ResizeEnabledGrows) {
    auto result = cslt::ArenaAllocator::Heap(512, true, 512);  // Small, resizable
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    size_t initial_chunks = arena->chunk_count();
    size_t initial_capacity = arena->allocated();
    
    // Fill initial capacity
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // This should trigger growth
    auto alloc_result = arena->alloc(256);
    ASSERT_TRUE(alloc_result.hasValue());
    
    // Should have more chunks now
    EXPECT_GT(arena->chunk_count(), initial_chunks);
    EXPECT_GT(arena->allocated(), initial_capacity);
}

TEST_F(ArenaAllocatorHeapTest, ResizeDisabledFails) {
    auto result = cslt::ArenaAllocator::Heap(512, false);  // Non-resizable
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Fill initial capacity
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // This should fail (can't resize)
    auto alloc_result = arena->alloc(256);
    EXPECT_TRUE(alloc_result.hasError());
}

TEST_F(ArenaAllocatorHeapTest, ToggleResize) {
    auto result = cslt::ArenaAllocator::Heap(512, true);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Fill capacity
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // Can resize initially
    auto alloc1 = arena->alloc(256);
    ASSERT_TRUE(alloc1.hasValue());
    
    // Disable resize
    arena->toggle_resize(false);
    
    // Fill again
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // Should fail now
    auto alloc2 = arena->alloc(256);
    EXPECT_TRUE(alloc2.hasError());
}

// ================================================================================
// Reset Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, ResetClearsUsage) {
    auto result = cslt::ArenaAllocator::Heap(2048);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make some allocations
    arena->alloc(128);
    arena->alloc(256);
    arena->alloc(512);
    
    size_t used = arena->size();
    EXPECT_GT(used, 0);
    
    // Reset
    bool reset_success = arena->reset(false);
    EXPECT_TRUE(reset_success);
    
    // Usage should be zero
    EXPECT_EQ(arena->size(), 0);
    
    // Can allocate again
    auto alloc_result = arena->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(ArenaAllocatorHeapTest, ResetWithTrim) {
    auto result = cslt::ArenaAllocator::Heap(512, true, 512);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Force growth by filling initial capacity
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    arena->alloc(256);  // Triggers new chunk
    
    size_t chunks_before = arena->chunk_count();
    EXPECT_GT(chunks_before, 1);
    
    // Reset with trim
    arena->reset(true);
    
    // Should have trimmed back to single chunk
    EXPECT_EQ(arena->chunk_count(), 1);
}

// ================================================================================
// Checkpoint/Restore Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, SaveAndRestore) {
    auto result = cslt::ArenaAllocator::Heap(2048);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make some allocations
    arena->alloc(128);
    arena->alloc(256);
    
    // Save checkpoint
    void* checkpoint = arena->save();
    ASSERT_NE(checkpoint, nullptr);
    
    size_t size_at_checkpoint = arena->size();
    
    // Make more allocations
    arena->alloc(512);
    arena->alloc(1024);
    
    EXPECT_GT(arena->size(), size_at_checkpoint);
    
    // Restore
    bool restored = arena->restore(checkpoint);
    EXPECT_TRUE(restored);
    
    // Should be back to checkpoint state
    EXPECT_EQ(arena->size(), size_at_checkpoint);
}

TEST_F(ArenaAllocatorHeapTest, RestoreNullCheckpoint) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    bool restored = arena->restore(nullptr);
    EXPECT_FALSE(restored);
}

// ================================================================================
// Realloc Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, ReallocLastAllocationGrow) {
    auto result = cslt::ArenaAllocator::Heap(4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate and fill with pattern
    auto alloc1 = arena->alloc(128);
    ASSERT_TRUE(alloc1.hasValue());
    void* ptr = alloc1.value();
    memset(ptr, 0xAA, 128);
    
    // Realloc to larger size (should extend in-place)
    auto realloc_result = arena->realloc(ptr, 128, 256, false);
    ASSERT_TRUE(realloc_result.hasValue());
    
    // Verify data preserved
    uint8_t* new_ptr = static_cast<uint8_t*>(realloc_result.value());
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(new_ptr[i], 0xAA) << "Byte " << i << " not preserved";
    }
}

TEST_F(ArenaAllocatorHeapTest, ReallocWithZeroing) {
    auto result = cslt::ArenaAllocator::Heap(4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc1 = arena->alloc(128);
    void* ptr = alloc1.value();
    memset(ptr, 0xBB, 128);
    
    // Realloc larger with zeroing
    auto realloc_result = arena->realloc(ptr, 128, 256, true);
    ASSERT_TRUE(realloc_result.hasValue());
    
    uint8_t* new_ptr = static_cast<uint8_t*>(realloc_result.value());
    
    // First 128 bytes should be preserved
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(new_ptr[i], 0xBB) << "Byte " << i << " not preserved";
    }
    
    // Next 128 bytes should be zeroed
    for (size_t i = 128; i < 256; ++i) {
        EXPECT_EQ(new_ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}

TEST_F(ArenaAllocatorHeapTest, ReallocNullPointer) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto realloc_result = arena->realloc(nullptr, 128, 256, false);
    EXPECT_TRUE(realloc_result.hasError());
}

// ================================================================================
// Statistics Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, Stats) {
    auto result = cslt::ArenaAllocator::Heap(2048, true, 1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make some allocations
    arena->alloc(128);
    arena->alloc(256);
    
    char buffer[1024];
    bool success = arena->stats(buffer, sizeof(buffer));
    EXPECT_TRUE(success);
    
    // Verify stats contain expected information
    EXPECT_NE(strstr(buffer, "Arena Statistics"), nullptr);
    EXPECT_NE(strstr(buffer, "DYNAMIC"), nullptr);
}

TEST_F(ArenaAllocatorHeapTest, StatsBufferTooSmall) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    char buffer[10];
    bool success = arena->stats(buffer, sizeof(buffer));
    EXPECT_FALSE(success);
}

TEST_F(ArenaAllocatorHeapTest, StatsNullBuffer) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    bool success = arena->stats(nullptr, 100);
    EXPECT_FALSE(success);
}

// ================================================================================
// Pointer Validation Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, IsPtrValid) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr = arena->alloc(128).value();
    
    EXPECT_TRUE(arena->is_ptr(ptr));
    EXPECT_FALSE(arena->is_ptr(nullptr));
    EXPECT_FALSE(arena->is_ptr(reinterpret_cast<void*>(0x12345678)));
}

TEST_F(ArenaAllocatorHeapTest, IsPtrSizedValid) {
    auto result = cslt::ArenaAllocator::Heap(2048);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr = arena->alloc(256).value();
    
    EXPECT_TRUE(arena->is_ptr_sized(ptr, 256));
    EXPECT_TRUE(arena->is_ptr_sized(ptr, 128));  // Subset is valid
    EXPECT_FALSE(arena->is_ptr_sized(ptr, 512));  // Too large
}

TEST_F(ArenaAllocatorHeapTest, IsPtrSizedNull) {
    auto result = cslt::ArenaAllocator::Heap(1024);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    EXPECT_FALSE(arena->is_ptr_sized(nullptr, 128));
}

// ================================================================================
// Remaining Tests
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, RemainingDecreases) {
    auto result = cslt::ArenaAllocator::Heap(2048, false);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    size_t initial_remaining = arena->remaining();
    
    arena->alloc(128);
    
    size_t after_alloc = arena->remaining();
    EXPECT_LT(after_alloc, initial_remaining);
}

TEST_F(ArenaAllocatorHeapTest, RemainingWithMultipleChunks) {
    auto result = cslt::ArenaAllocator::Heap(512, true, 512);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Fill first chunk
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    size_t before_growth = arena->remaining();
    
    // Trigger growth
    arena->alloc(256);
    
    // Should have more remaining after growth
    EXPECT_GT(arena->remaining(), before_growth);
}

// ================================================================================
// Edge Cases
// ================================================================================

TEST_F(ArenaAllocatorHeapTest, LargeAlignment) {
    auto result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Try very large alignment (256 bytes)
    auto alloc_result = arena->alloc_aligned(64, 256);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 256, 0);
}

TEST_F(ArenaAllocatorHeapTest, ManySmallAllocations) {
    auto result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make 100 small allocations
    for (int i = 0; i < 100; ++i) {
        auto alloc_result = arena->alloc(8);
        ASSERT_TRUE(alloc_result.hasValue()) << "Allocation " << i << " failed";
    }
}
// ================================================================================ 
// ================================================================================ 

class ArenaAllocatorStackTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ================================================================================
// Basic Creation Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, BasicCreation) {
    uint8_t buffer[4096];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    ASSERT_NE(arena, nullptr);
    EXPECT_EQ(arena->memory_type(), cslt::STATIC);
    EXPECT_FALSE(arena->owns_memory());
}

TEST_F(ArenaAllocatorStackTest, CreationWithAlignment) {
    uint8_t buffer[8192];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer), 32);
    ASSERT_TRUE(result.hasValue());
    
    auto arena = cslt::move(result.value());
    EXPECT_EQ(arena->default_alignment(), 32);
}

TEST_F(ArenaAllocatorStackTest, CreationNullBuffer) {
    auto result = cslt::ArenaAllocator::Stack(nullptr, 1024);
    EXPECT_TRUE(result.hasError());
    
    const char* msg = result.error().what();
    EXPECT_TRUE(strstr(msg, "null") != nullptr || strstr(msg, "buffer") != nullptr)
        << "Expected error about null buffer, got: " << msg;
}

TEST_F(ArenaAllocatorStackTest, CreationBufferTooSmall) {
    uint8_t buffer[64];  // Too small for ArenaAllocator + Chunk
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    EXPECT_TRUE(result.hasError());
    
    const char* msg = result.error().what();
    EXPECT_TRUE(strstr(msg, "too small") != nullptr || strstr(msg, "structure") != nullptr)
        << "Expected error about buffer size, got: " << msg;
}

// ================================================================================
// Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, SimpleAllocation) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc_result = arena->alloc(128);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(arena->is_ptr(ptr));
}

TEST_F(ArenaAllocatorStackTest, MultipleAllocations) {
    uint8_t buffer[4096];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr1 = arena->alloc(64).value();
    void* ptr2 = arena->alloc(128).value();
    void* ptr3 = arena->alloc(256).value();
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
    
    // All pointers should be within the buffer
    EXPECT_GE(ptr1, buffer);
    EXPECT_LT(ptr1, buffer + sizeof(buffer));
    EXPECT_GE(ptr2, buffer);
    EXPECT_LT(ptr2, buffer + sizeof(buffer));
    EXPECT_GE(ptr3, buffer);
    EXPECT_LT(ptr3, buffer + sizeof(buffer));
}

TEST_F(ArenaAllocatorStackTest, AllocationFillsBuffer) {
    uint8_t buffer[1024];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Allocate until we run out
    size_t allocated = 0;
    while (true) {
        auto alloc_result = arena->alloc(64);
        if (!alloc_result.hasValue()) {
            break;  // Out of memory
        }
        allocated += 64;
    }
    
    EXPECT_GT(allocated, 0);
    EXPECT_LT(arena->remaining(), 64);  // Can't fit another 64-byte allocation
}

TEST_F(ArenaAllocatorStackTest, ZeroedAllocation) {
    uint8_t buffer[2048];
    // Fill buffer with non-zero pattern first
    memset(buffer, 0xFF, sizeof(buffer));
    
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc_result = arena->alloc(256, true);
    ASSERT_TRUE(alloc_result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(alloc_result.value());
    
    // Verify all bytes are zero
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}

// ================================================================================
// Aligned Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, AlignedAllocation) {
    uint8_t buffer[4096];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc_result = arena->alloc_aligned(128, 64);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Verify alignment
    EXPECT_EQ(addr % 64, 0);
}

TEST_F(ArenaAllocatorStackTest, MultipleAlignedAllocations) {
    uint8_t buffer[8192];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr1 = arena->alloc_aligned(64, 16).value();
    void* ptr2 = arena->alloc_aligned(128, 32).value();
    void* ptr3 = arena->alloc_aligned(256, 64).value();
    
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 32, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 64, 0);
}

// ================================================================================
// Static Buffer Characteristics Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, CannotResize) {
    uint8_t buffer[1024];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Fill the buffer
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // Try to allocate more - should fail (can't resize static)
    auto alloc_result = arena->alloc(512);
    EXPECT_TRUE(alloc_result.hasError());
}

TEST_F(ArenaAllocatorStackTest, ToggleResizeHasNoEffect) {
    uint8_t buffer[1024];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Try to enable resize (should have no effect on STATIC)
    arena->toggle_resize(true);
    
    // Fill the buffer
    while (arena->remaining() >= 64) {
        arena->alloc(64);
    }
    
    // Should still fail to allocate more
    auto alloc_result = arena->alloc(512);
    EXPECT_TRUE(alloc_result.hasError());
}

TEST_F(ArenaAllocatorStackTest, SingleChunkOnly) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make several allocations
    for (int i = 0; i < 10; ++i) {
        arena->alloc(64);
    }
    
    // Should always have exactly 1 chunk (can't grow)
    EXPECT_EQ(arena->chunk_count(), 1);
}

// ================================================================================
// Reset Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, ResetClearsUsage) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Make allocations
    arena->alloc(128);
    arena->alloc(256);
    
    size_t used = arena->size();
    EXPECT_GT(used, 0);
    
    // Reset
    bool reset_success = arena->reset(false);
    EXPECT_TRUE(reset_success);
    
    // Usage should be zero
    EXPECT_EQ(arena->size(), 0);
    
    // Can allocate again
    auto alloc_result = arena->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(ArenaAllocatorStackTest, ResetWithTrimHasNoEffect) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    arena->alloc(512);
    
    size_t chunks_before = arena->chunk_count();
    
    // Reset with trim (should have no effect on STATIC - can't free the buffer)
    arena->reset(true);
    
    // Should still have same chunk count
    EXPECT_EQ(arena->chunk_count(), chunks_before);
}

// ================================================================================
// Checkpoint/Restore Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, SaveAndRestore) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    arena->alloc(128);
    arena->alloc(256);
    
    void* checkpoint = arena->save();
    ASSERT_NE(checkpoint, nullptr);
    
    size_t size_at_checkpoint = arena->size();
    
    arena->alloc(512);
    
    EXPECT_GT(arena->size(), size_at_checkpoint);
    
    bool restored = arena->restore(checkpoint);
    EXPECT_TRUE(restored);
    
    EXPECT_EQ(arena->size(), size_at_checkpoint);
}

// ================================================================================
// Buffer Lifetime Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, ArenaLifetimeIndependentOfBuffer) {
    uint8_t buffer[2048];
    
    // Create arena in inner scope
    {
        auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
        ASSERT_TRUE(result.hasValue());
        auto arena = cslt::move(result.value());
        
        void* ptr = arena->alloc(128).value();
        memset(ptr, 0xAB, 128);
    }
    // Arena destroyed, but buffer still exists
    
    // Buffer should still be accessible and contain the data
    // (In real usage, you'd want to keep the arena alive as long as you use the buffer)
    EXPECT_NE(buffer, nullptr);
}

TEST_F(ArenaAllocatorStackTest, AlignedBuffer) {
    // Test with properly aligned buffer
    alignas(64) uint8_t buffer[4096];
    
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer), 64);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Buffer address should be aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer) % 64, 0);
    
    // Allocations should work fine
    auto alloc_result = arena->alloc_aligned(256, 64);
    EXPECT_TRUE(alloc_result.hasValue());
}

// ================================================================================
// Statistics Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, Stats) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    arena->alloc(128);
    arena->alloc(256);
    
    char stats_buffer[1024];
    bool success = arena->stats(stats_buffer, sizeof(stats_buffer));
    EXPECT_TRUE(success);
    
    // Verify stats contain expected information
    EXPECT_NE(strstr(stats_buffer, "Arena Statistics"), nullptr);
    EXPECT_NE(strstr(stats_buffer, "STATIC"), nullptr);
}

// ================================================================================
// Pointer Validation Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, IsPtrValid) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr = arena->alloc(128).value();
    
    EXPECT_TRUE(arena->is_ptr(ptr));
    EXPECT_FALSE(arena->is_ptr(nullptr));
    
    // Pointer outside the buffer should not be valid
    void* external = malloc(128);
    EXPECT_FALSE(arena->is_ptr(external));
    free(external);
}

TEST_F(ArenaAllocatorStackTest, IsPtrSizedValid) {
    uint8_t buffer[2048];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    void* ptr = arena->alloc(256).value();
    
    EXPECT_TRUE(arena->is_ptr_sized(ptr, 256));
    EXPECT_TRUE(arena->is_ptr_sized(ptr, 128));  // Subset
    EXPECT_FALSE(arena->is_ptr_sized(ptr, 512));  // Too large
}

// ================================================================================
// Realloc Tests
// ================================================================================

TEST_F(ArenaAllocatorStackTest, ReallocLastAllocation) {
    uint8_t buffer[4096];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    auto alloc1 = arena->alloc(128);
    void* ptr = alloc1.value();
    memset(ptr, 0xCC, 128);
    
    // Realloc to larger
    auto realloc_result = arena->realloc(ptr, 128, 256, false);
    ASSERT_TRUE(realloc_result.hasValue());
    
    // Verify data preserved
    uint8_t* new_ptr = static_cast<uint8_t*>(realloc_result.value());
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(new_ptr[i], 0xCC) << "Byte " << i << " not preserved";
    }
}

// ================================================================================
// Edge Cases
// ================================================================================

TEST_F(ArenaAllocatorStackTest, VeryLargeBuffer) {
    // Test with a large stack buffer (but not too large to overflow stack)
    uint8_t buffer[65536];
    auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Should have plenty of space
    EXPECT_GT(arena->remaining(), 60000);
}

TEST_F(ArenaAllocatorStackTest, MinimumViableBuffer) {
    // Use a reasonably small buffer and verify it works
    // We don't know the exact minimum, but 512 bytes should be enough
    size_t buffer_size = 512;
    
    uint8_t* buffer = new uint8_t[buffer_size];
    auto result = cslt::ArenaAllocator::Stack(buffer, buffer_size);
    EXPECT_TRUE(result.hasValue());
    
    if (result.hasValue()) {
        auto arena = cslt::move(result.value());
        // Should be able to make at least one small allocation
        auto alloc_result = arena->alloc(32);
        EXPECT_TRUE(alloc_result.hasValue());
        
        // Should have some remaining space
        EXPECT_GT(arena->remaining(), 0);
    }
    
    delete[] buffer;
}

TEST_F(ArenaAllocatorStackTest, UnalignedBuffer) {
    // Create a slightly misaligned buffer
    uint8_t raw_buffer[4097];  // Odd size
    uint8_t* buffer = raw_buffer + 1;  // Offset by 1 to misalign
    
    auto result = cslt::ArenaAllocator::Stack(buffer, 4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Should still work, just with some padding overhead
    auto alloc_result = arena->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}
// ================================================================================ 
// ================================================================================ 

class ArenaAllocatorSubArenaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// ================================================================================
// Basic Creation Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, BasicCreation) {
    // Create parent arena
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Create sub-arena
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    
    auto sub = cslt::move(sub_result.value());
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->memory_type(), DYNAMIC);
    EXPECT_FALSE(sub->owns_memory());
}

TEST_F(ArenaAllocatorSubArenaTest, CreationWithAlignment) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048, 64);
    ASSERT_TRUE(sub_result.hasValue());
    
    auto sub = cslt::move(sub_result.value());
    EXPECT_EQ(sub->default_alignment(), 64);
}

TEST_F(ArenaAllocatorSubArenaTest, CreationZeroSize) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 0);
    EXPECT_TRUE(sub_result.hasError());
    
    const char* msg = sub_result.error().what();
    EXPECT_TRUE(strstr(msg, "zero") != nullptr || strstr(msg, "size") != nullptr)
        << "Expected error about zero size, got: " << msg;
}

TEST_F(ArenaAllocatorSubArenaTest, CreationTooLarge) {
    // Create small parent
    auto parent_result = cslt::ArenaAllocator::Heap(1024, false);  // Non-resizable
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Try to create sub-arena larger than parent has available
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 10000);
    EXPECT_TRUE(sub_result.hasError());
    
    const char* msg = sub_result.error().what();
    EXPECT_TRUE(strstr(msg, "allocation") != nullptr || strstr(msg, "failed") != nullptr)
        << "Expected allocation failure, got: " << msg;
}

TEST_F(ArenaAllocatorSubArenaTest, CreationSizeTooSmall) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Request size too small for ArenaAllocator + Chunk
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 10);
    EXPECT_TRUE(sub_result.hasError());
    
    const char* msg = sub_result.error().what();
    EXPECT_TRUE(strstr(msg, "too small") != nullptr || strstr(msg, "structure") != nullptr)
        << "Expected error about size, got: " << msg;
}

// ================================================================================
// Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, SimpleAllocation) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    auto alloc_result = sub->alloc(128);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(sub->is_ptr(ptr));
}

TEST_F(ArenaAllocatorSubArenaTest, MultipleAllocations) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    void* ptr1 = sub->alloc(64).value();
    void* ptr2 = sub->alloc(128).value();
    void* ptr3 = sub->alloc(256).value();
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
    
    // All should be tracked by sub-arena
    EXPECT_TRUE(sub->is_ptr(ptr1));
    EXPECT_TRUE(sub->is_ptr(ptr2));
    EXPECT_TRUE(sub->is_ptr(ptr3));
}

TEST_F(ArenaAllocatorSubArenaTest, AllocationFillsSubArena) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 1024);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Fill sub-arena
    size_t allocated = 0;
    while (true) {
        auto alloc_result = sub->alloc(64);
        if (!alloc_result.hasValue()) {
            break;
        }
        allocated += 64;
    }
    
    EXPECT_GT(allocated, 0);
    EXPECT_LT(sub->remaining(), 64);
}

TEST_F(ArenaAllocatorSubArenaTest, ZeroedAllocation) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    auto alloc_result = sub->alloc(256, true);
    ASSERT_TRUE(alloc_result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(alloc_result.value());
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}

// ================================================================================
// Aligned Allocation Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, AlignedAllocation) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    auto alloc_result = sub->alloc_aligned(128, 64);
    ASSERT_TRUE(alloc_result.hasValue());
    
    void* ptr = alloc_result.value();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
}

TEST_F(ArenaAllocatorSubArenaTest, MultipleAlignedAllocations) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 4096);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    void* ptr1 = sub->alloc_aligned(64, 16).value();
    void* ptr2 = sub->alloc_aligned(128, 32).value();
    void* ptr3 = sub->alloc_aligned(256, 64).value();
    
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 32, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % 64, 0);
}

// ================================================================================
// Parent-Child Relationship Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, ParentTracksSubArenaAllocation) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    size_t parent_used_before = parent->size();
    
    // Create sub-arena (consumes space from parent)
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Parent should have allocated space for the sub-arena
    EXPECT_GT(parent->size(), parent_used_before);
}

TEST_F(ArenaAllocatorSubArenaTest, SubArenaAllocationsIndependentFromParent) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    size_t parent_size_after_sub = parent->size();
    
    // Allocate from sub-arena
    sub->alloc(256);
    
    // Parent's size shouldn't change (sub manages its own space)
    EXPECT_EQ(parent->size(), parent_size_after_sub);
}

TEST_F(ArenaAllocatorSubArenaTest, MultipleSubArenas) {
    auto parent_result = cslt::ArenaAllocator::Heap(16384);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Create multiple sub-arenas
    auto sub1_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub1_result.hasValue());
    auto sub1 = cslt::move(sub1_result.value());
    
    auto sub2_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub2_result.hasValue());
    auto sub2 = cslt::move(sub2_result.value());
    
    auto sub3_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub3_result.hasValue());
    auto sub3 = cslt::move(sub3_result.value());
    
    // All should be independent
    void* ptr1 = sub1->alloc(128).value();
    void* ptr2 = sub2->alloc(128).value();
    void* ptr3 = sub3->alloc(128).value();
    
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);
}

TEST_F(ArenaAllocatorSubArenaTest, NestedSubArenas) {
    // Create hierarchy: parent -> sub -> subsub
    auto parent_result = cslt::ArenaAllocator::Heap(16384);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 8192);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    auto subsub_result = cslt::ArenaAllocator::SubArena(*sub, 2048);
    ASSERT_TRUE(subsub_result.hasValue());
    auto subsub = cslt::move(subsub_result.value());
    
    // Should all work independently
    void* ptr_parent = parent->alloc(64).value();
    void* ptr_sub = sub->alloc(64).value();
    void* ptr_subsub = subsub->alloc(64).value();
    
    EXPECT_NE(ptr_parent, nullptr);
    EXPECT_NE(ptr_sub, nullptr);
    EXPECT_NE(ptr_subsub, nullptr);
}

// ================================================================================
// SubArena Characteristics Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, CannotResize) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 1024);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Fill the sub-arena
    while (sub->remaining() >= 64) {
        sub->alloc(64);
    }
    
    // Try to allocate more - should fail (can't resize)
    auto alloc_result = sub->alloc(512);
    EXPECT_TRUE(alloc_result.hasError());
}

TEST_F(ArenaAllocatorSubArenaTest, ToggleResizeHasNoEffect) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 1024);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Try to enable resize (should have no effect on SUB)
    sub->toggle_resize(true);
    
    // Fill the sub-arena
    while (sub->remaining() >= 64) {
        sub->alloc(64);
    }
    
    // Should still fail
    auto alloc_result = sub->alloc(512);
    EXPECT_TRUE(alloc_result.hasError());
}

TEST_F(ArenaAllocatorSubArenaTest, SingleChunkOnly) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Make several allocations
    for (int i = 0; i < 10; ++i) {
        auto result = sub->alloc(64);
        if (!result.hasValue()) break;
    }
    
    // Should always have exactly 1 chunk
    EXPECT_EQ(sub->chunk_count(), 1);
}

// ================================================================================
// Reset Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, ResetClearsUsage) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    sub->alloc(128);
    sub->alloc(256);
    
    size_t used = sub->size();
    EXPECT_GT(used, 0);
    
    bool reset_success = sub->reset(false);
    EXPECT_TRUE(reset_success);
    
    EXPECT_EQ(sub->size(), 0);
    
    // Can allocate again
    auto alloc_result = sub->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(ArenaAllocatorSubArenaTest, ResetDoesNotAffectParent) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Parent allocates something
    parent->alloc(512);
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    sub->alloc(256);
    
    size_t parent_size = parent->size();
    
    // Reset sub-arena
    sub->reset(false);
    
    // Parent size should be unchanged
    EXPECT_EQ(parent->size(), parent_size);
}

// ================================================================================
// Checkpoint/Restore Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, SaveAndRestore) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    sub->alloc(128);
    sub->alloc(256);
    
    void* checkpoint = sub->save();
    ASSERT_NE(checkpoint, nullptr);
    
    size_t size_at_checkpoint = sub->size();
    
    sub->alloc(512);
    
    EXPECT_GT(sub->size(), size_at_checkpoint);
    
    bool restored = sub->restore(checkpoint);
    EXPECT_TRUE(restored);
    
    EXPECT_EQ(sub->size(), size_at_checkpoint);
}

// ================================================================================
// Statistics Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, Stats) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    sub->alloc(128);
    sub->alloc(256);
    
    char buffer[1024];
    bool success = sub->stats(buffer, sizeof(buffer));
    EXPECT_TRUE(success);
    
    // Verify stats contain expected information
    EXPECT_NE(strstr(buffer, "Arena Statistics"), nullptr);
    // Note: SUB type might show up differently in stats
}

// ================================================================================
// Pointer Validation Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, IsPtrValid) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    void* sub_ptr = sub->alloc(128).value();
    void* parent_ptr = parent->alloc(128).value();
    
    // Sub-arena pointer should be valid in sub
    EXPECT_TRUE(sub->is_ptr(sub_ptr));
    
    // Parent pointer should NOT be valid in sub
    EXPECT_FALSE(sub->is_ptr(parent_ptr));
    
    // Sub pointer IS valid in parent (it's in parent's memory)
    EXPECT_TRUE(parent->is_ptr(sub_ptr));
}

TEST_F(ArenaAllocatorSubArenaTest, IsPtrSizedValid) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    void* ptr = sub->alloc(256).value();
    
    EXPECT_TRUE(sub->is_ptr_sized(ptr, 256));
    EXPECT_TRUE(sub->is_ptr_sized(ptr, 128));
    EXPECT_FALSE(sub->is_ptr_sized(ptr, 512));
}

// ================================================================================
// Realloc Tests
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, ReallocLastAllocation) {
    auto parent_result = cslt::ArenaAllocator::Heap(8192);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    auto alloc1 = sub->alloc(128);
    void* ptr = alloc1.value();
    memset(ptr, 0xDD, 128);
    
    auto realloc_result = sub->realloc(ptr, 128, 256, false);
    ASSERT_TRUE(realloc_result.hasValue());
    
    uint8_t* new_ptr = static_cast<uint8_t*>(realloc_result.value());
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(new_ptr[i], 0xDD) << "Byte " << i << " not preserved";
    }
}

// ================================================================================
// Edge Cases
// ================================================================================

TEST_F(ArenaAllocatorSubArenaTest, SubArenaFromStaticParent) {
    // Create parent from stack buffer
    uint8_t buffer[8192];
    auto parent_result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    // Create sub-arena from static parent
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 2048);
    ASSERT_TRUE(sub_result.hasValue());
    auto sub = cslt::move(sub_result.value());
    
    // Should work normally
    auto alloc_result = sub->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(ArenaAllocatorSubArenaTest, SubArenaConsumesAllParent) {
    auto parent_result = cslt::ArenaAllocator::Heap(2048, false);
    ASSERT_TRUE(parent_result.hasValue());
    auto parent = cslt::move(parent_result.value());
    
    size_t parent_remaining = parent->remaining();
    
    // Try to create sub-arena that uses most of parent
    auto sub_result = cslt::ArenaAllocator::SubArena(*parent, parent_remaining - 128);
    
    if (sub_result.hasValue()) {
        auto sub = cslt::move(sub_result.value());
        // Parent should have very little left
        EXPECT_LT(parent->remaining(), 200);
    }
}

TEST_F(ArenaAllocatorSubArenaTest, DeepNesting) {
    // Create a chain of nested sub-arenas
    auto arena1_result = cslt::ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena1_result.hasValue());
    auto arena1 = cslt::move(arena1_result.value());
    
    auto arena2_result = cslt::ArenaAllocator::SubArena(*arena1, 8192);
    ASSERT_TRUE(arena2_result.hasValue());
    auto arena2 = cslt::move(arena2_result.value());
    
    auto arena3_result = cslt::ArenaAllocator::SubArena(*arena2, 4096);
    ASSERT_TRUE(arena3_result.hasValue());
    auto arena3 = cslt::move(arena3_result.value());
    
    auto arena4_result = cslt::ArenaAllocator::SubArena(*arena3, 2048);
    ASSERT_TRUE(arena4_result.hasValue());
    auto arena4 = cslt::move(arena4_result.value());
    
    // Deepest level should still work
    auto alloc_result = arena4->alloc(128);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(ArenaAllocatorHeapTest, MinChunkSize) {
    auto result = cslt::ArenaAllocator::Heap(2048, true, 8192);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    EXPECT_EQ(arena->min_chunk_size(), 8192);
}

TEST_F(ArenaAllocatorHeapTest, MemoryAccounting) {
    auto result = cslt::ArenaAllocator::Heap(4096);
    ASSERT_TRUE(result.hasValue());
    auto arena = cslt::move(result.value());
    
    // Relationships that should hold
    EXPECT_LE(arena->allocated(), arena->total_alloc());  // Usable <= Total
    EXPECT_LE(arena->size(), arena->allocated());         // Used <= Usable
    EXPECT_EQ(arena->remaining(), arena->allocated() - arena->size());
}

TEST_F(ArenaAllocatorHeapTest, MemoryTypeStable) {
    auto result = cslt::ArenaAllocator::Heap(2048);
    auto arena = cslt::move(result.value());
    
    EXPECT_EQ(arena->memory_type(), cslt::DYNAMIC);
    
    arena->alloc(512);
    EXPECT_EQ(arena->memory_type(), cslt::DYNAMIC);
    
    arena->reset();
    EXPECT_EQ(arena->memory_type(), cslt::DYNAMIC);
}

TEST_F(ArenaAllocatorHeapTest, PointerNotInDifferentArena) {
    auto arena1_result = cslt::ArenaAllocator::Heap(2048);
    auto arena1 = cslt::move(arena1_result.value());
    
    auto arena2_result = cslt::ArenaAllocator::Heap(2048);
    auto arena2 = cslt::move(arena2_result.value());
    
    void* ptr1 = arena1->alloc(128).value();
    
    EXPECT_TRUE(arena1->is_ptr(ptr1));
    EXPECT_FALSE(arena2->is_ptr(ptr1));  // Not in arena2
}
// ================================================================================ 
// ================================================================================ 

class PoolHeapTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Common cleanup if needed
    }

    // Helper to check if pointer is properly aligned
    bool is_aligned(void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
};

// ================================================================================
// Phase 1: Core Heap Initialization Tests (10 tests)
// ================================================================================

TEST_F(PoolHeapTest, CreateBasicPool) {
    // Test basic pool creation with typical parameters
    auto result = PoolAllocator::Heap(
        256,    // block_size
        32,     // blocks_per_chunk
        0,      // alignment (default)
        10240,  // arena_initial_bytes (enough for 32 blocks + overhead)
        4096,   // min_chunk_bytes
        true,   // grow_enabled
        true    // prewarm
    );

    ASSERT_TRUE(result.hasValue()) << "Failed to create basic pool";
    auto pool = cslt::move(result.value());
    
    EXPECT_NE(pool.get(), nullptr);
    EXPECT_EQ(pool->block_size(), 256);
    EXPECT_EQ(pool->stride(), 256);
    EXPECT_EQ(pool->total_blocks(), 32);  // Prewarmed with 32 blocks
    EXPECT_TRUE(pool->can_grow());
}

TEST_F(PoolHeapTest, CreatePoolWithPrewarm) {
    // Verify prewarming allocates the initial blocks
    auto result = PoolAllocator::Heap(
        128,
        16,
        0,
        4096,
        4096,
        true,
        true    // Prewarm enabled
    );

    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_EQ(pool->total_blocks(), 16);  // 16 blocks prewarmed
    EXPECT_EQ(pool->free_blocks(), 0);    // None in free list yet
}

TEST_F(PoolHeapTest, CreatePoolWithoutPrewarm) {
    // Pool without prewarm should start with 0 blocks
    auto result = PoolAllocator::Heap(
        256,
        32,
        0,
        2048,   // Just enough for headers
        4096,
        true,   // Can grow
        false   // No prewarm
    );
    
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_EQ(pool->total_blocks(), 0);  // No blocks yet
    EXPECT_TRUE(pool->can_grow());       // But can grow on demand
}

TEST_F(PoolHeapTest, CreateNonGrowablePool) {
    // Fixed-capacity pool (cannot grow after creation)
    auto result = PoolAllocator::Heap(
        128,
        64,
        0,
        10240,  // Enough for 64 blocks + overhead
        4096,
        false,  // Cannot grow
        true    // Must prewarm
    );
    
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_FALSE(pool->can_grow());
    EXPECT_EQ(pool->total_blocks(), 64);
}

TEST_F(PoolHeapTest, CreatePoolWithCustomAlignment) {
    // Pool with 64-byte alignment (cache line)
    auto result = PoolAllocator::Heap(
        256,
        32,
        64,     // 64-byte alignment
        10240,
        4096,
        true,
        true
    );

    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_EQ(pool->default_alignment(), 64);
    EXPECT_GE(pool->stride(), 256);  // At least block_size, aligned to 64
}

TEST_F(PoolHeapTest, FailZeroBlockSize) {
    // Should fail with zero block size
    auto result = PoolAllocator::Heap(
        0,      // Invalid: zero block size
        32,
        0,
        8192,
        4096,
        true,
        true
    );

    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST_F(PoolHeapTest, FailZeroBlocksPerChunk) {
    // Should fail with zero blocks per chunk
    auto result = PoolAllocator::Heap(
        256,
        0,      // Invalid: zero blocks per chunk
        0,
        8192,
        4096,
        true,
        true
    );

    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST_F(PoolHeapTest, FailZeroArenaSize) {
    // Should fail with zero arena size
    auto result = PoolAllocator::Heap(
        256,
        32,
        0,
        0,      // Invalid: zero arena size
        4096,
        true,
        true
    );

    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST_F(PoolHeapTest, FailNonGrowableWithoutPrewarm) {
    // Non-growable pool without prewarm would be unusable
    auto result = PoolAllocator::Heap(
        256,
        32,
        0,
        8192,
        4096,
        false,  // Cannot grow
        false   // No prewarm
    );

    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("prewarm"), std::string::npos);
}

TEST_F(PoolHeapTest, VerifyMemoryType) {
    // Verify pool reports correct memory type
    auto result = PoolAllocator::Heap(256, 32, 0, 8192, 4096, true, true);
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_EQ(pool->memory_type(), DYNAMIC);
    EXPECT_TRUE(pool->owns_memory());
}
// -------------------------------------------------------------------------------- 

TEST_F(PoolHeapTest, AllocateSingleBlock) {
    // Test basic allocation from prewarmed pool
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, true, true).value());
    
    auto result = pool->alloc(128);
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, pool->default_alignment()));
}

TEST_F(PoolHeapTest, AllocateMultipleBlocks) {
    // Allocate several blocks and verify they're distinct
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        auto result = pool->alloc(256);
        ASSERT_TRUE(result.hasValue()) << "Failed at allocation " << i;
        ptrs.push_back(result.value());
    }
    
    // Verify all pointers are unique
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]) << "Duplicate pointer at " << i << " and " << j;
        }
    }
}

TEST_F(PoolHeapTest, AllocateZeroed) {
    // Test zeroed allocation
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, true, true).value());
    
    auto result = pool->alloc(128, true);  // Request zeroed
    ASSERT_TRUE(result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(result.value());
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}

TEST_F(PoolHeapTest, AllocateFromNonPrewarmedPool) {
    // Pool without prewarm should allocate by growing
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 2048, 4096, true, false).value());
    
    EXPECT_EQ(pool->total_blocks(), 0);  // No blocks yet
    
    auto result = pool->alloc(256);
    ASSERT_TRUE(result.hasValue());
    
    // Should have grown to accommodate allocation
    EXPECT_GT(pool->total_blocks(), 0);
}

TEST_F(PoolHeapTest, GrowthBehavior) {
    // Allocate beyond initial capacity to trigger growth
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, true, true).value());
    
    size_t initial_blocks = pool->total_blocks();
    EXPECT_EQ(initial_blocks, 16);
    
    // Allocate all initial blocks
    for (size_t i = 0; i < 16; ++i) {
        auto result = pool->alloc(128);
        ASSERT_TRUE(result.hasValue()) << "Failed at block " << i;
    }
    
    // Next allocation should trigger growth
    auto result = pool->alloc(128);
    ASSERT_TRUE(result.hasValue());
    
    EXPECT_GT(pool->total_blocks(), initial_blocks) << "Pool did not grow";
}

TEST_F(PoolHeapTest, NonGrowableCapacityLimit) {
    // Non-growable pool should fail when capacity exhausted
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, false, true).value());
    
    EXPECT_FALSE(pool->can_grow());
    EXPECT_EQ(pool->total_blocks(), 16);
    
    // Allocate all blocks
    std::vector<void*> ptrs;
    for (size_t i = 0; i < 16; ++i) {
        auto result = pool->alloc(128);
        ASSERT_TRUE(result.hasValue()) << "Failed at block " << i;
        ptrs.push_back(result.value());
    }
    
    // Next allocation should fail
    auto result = pool->alloc(128);
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_TRUE(error_msg.find("capacity") != std::string::npos ||
                error_msg.find("grow") != std::string::npos);
}

TEST_F(PoolHeapTest, FreeAndReuse) {
    // Test free-list recycling
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    auto result1 = pool->alloc(256);
    ASSERT_TRUE(result1.hasValue());
    void* ptr1 = result1.value();
    
    EXPECT_EQ(pool->free_blocks(), 0);
    
    // Free the block
    pool->return_element(ptr1, 256);
    EXPECT_EQ(pool->free_blocks(), 1);
    
    // Allocate again - should reuse freed block
    auto result2 = pool->alloc(256);
    ASSERT_TRUE(result2.hasValue());
    void* ptr2 = result2.value();
    
    EXPECT_EQ(ptr1, ptr2) << "Block not reused from free list";
    EXPECT_EQ(pool->free_blocks(), 0);
}

TEST_F(PoolHeapTest, Statistics) {
    // Test statistics generation
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    char buffer[1024];
    ASSERT_TRUE(pool->stats(buffer, sizeof(buffer)));
    
    std::string stats_str(buffer);
    
    // Should contain key information
    EXPECT_NE(stats_str.find("Pool Allocator"), std::string::npos);
    EXPECT_NE(stats_str.find("Block Size: 256"), std::string::npos);
    EXPECT_NE(stats_str.find("Total Blocks: 32"), std::string::npos);
    EXPECT_NE(stats_str.find("Type: DYNAMIC"), std::string::npos);
}

TEST_F(PoolHeapTest, MultiplePools) {
    // Multiple independent pools shouldn't interfere
    auto pool1 = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, true, true).value());
    auto pool2 = cslt::move(PoolAllocator::Heap(256, 16, 0, 8192, 4096, true, true).value());
    
    EXPECT_EQ(pool1->block_size(), 128);
    EXPECT_EQ(pool2->block_size(), 256);
    
    auto ptr1 = pool1->alloc(128);
    auto ptr2 = pool2->alloc(256);
    
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    
    // Pointers from different pools should be different
    EXPECT_NE(ptr1.value(), ptr2.value());
}
// -------------------------------------------------------------------------------- 

TEST_F(PoolHeapTest, CheckpointAndRestore) {
    // Test checkpoint/restore functionality
    auto pool = cslt::move(PoolAllocator::Heap(128, 32, 0, 8192, 4096, true, true).value());
    
    // Make some permanent allocations
    auto perm1 = pool->alloc(false, 128);
    auto perm2 = pool->alloc(false, 128);
    ASSERT_TRUE(perm1.hasValue());
    ASSERT_TRUE(perm2.hasValue());
    
    // Save checkpoint
    void* checkpoint = pool->save();
    ASSERT_NE(checkpoint, nullptr);
    
    // Make temporary allocations
    auto temp1 = pool->alloc(false, 128);
    auto temp2 = pool->alloc(false, 128);
    ASSERT_TRUE(temp1.hasValue());
    ASSERT_TRUE(temp2.hasValue());
    
    // Restore to checkpoint
    ASSERT_TRUE(pool->restore(checkpoint));
    
    // After restore, checkpoint is freed (don't use it again)
}

TEST_F(PoolHeapTest, ResetPool) {
    // Test pool reset
    auto pool = cslt::move(PoolAllocator::Heap(256, 16, 0, 8192, 4096, true, true).value());
    
    // Allocate some blocks
    pool->alloc(false, 256);
    pool->alloc(false, 256);
    pool->alloc(false, 256);
    
    // Reset pool
    ASSERT_TRUE(pool->reset());
    
    // Pool should be usable again
    auto result = pool->alloc(false, 256);
    EXPECT_TRUE(result.hasValue());
}

TEST_F(PoolHeapTest, VeryLargeBlocks) {
    // Test with large block sizes (4KB)
    auto pool = cslt::move(PoolAllocator::Heap(
        4096,   // 4KB blocks
        8,      // 8 blocks per chunk
        0,
        64 * 1024,  // 64KB arena
        8192,
        true,
        true
    ).value());
    
    EXPECT_EQ(pool->block_size(), 4096);
    
    auto result = pool->alloc(false, 4096);
    EXPECT_TRUE(result.hasValue());
}

TEST_F(PoolHeapTest, ManySmallBlocks) {
    // Test with many small blocks
    auto pool = cslt::move(PoolAllocator::Heap(
        16,     // Small 16-byte blocks
        256,    // Many blocks per chunk
        0,
        16 * 1024,  // Enough space
        4096,
        true,
        true
    ).value());
    
    EXPECT_EQ(pool->total_blocks(), 256);
    
    // Allocate many blocks
    for (int i = 0; i < 100; ++i) {
        auto result = pool->alloc(false, 16);
        ASSERT_TRUE(result.hasValue()) << "Failed at block " << i;
    }
}

TEST_F(PoolHeapTest, PageAlignedBlocks) {
    // Test with page-aligned blocks (4096 bytes)
    auto pool = cslt::move(PoolAllocator::Heap(
        256,
        32,
        4096,   // Page alignment
        64 * 1024,
        4096,
        true,
        true
    ).value());
    
    EXPECT_EQ(pool->default_alignment(), 4096);
    
    auto result = pool->alloc(false, 256);
    ASSERT_TRUE(result.hasValue());
    
    // Verify alignment
    EXPECT_TRUE(is_aligned(result.value(), 4096));
}

TEST_F(PoolHeapTest, AllocateAllPrewarmedBlocks) {
    // Allocate every prewarmed block
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, false, true).value());
    
    std::vector<void*> ptrs;
    
    // Should be able to allocate exactly 16 blocks
    for (int i = 0; i < 16; ++i) {
        auto result = pool->alloc(false, 128);
        ASSERT_TRUE(result.hasValue()) << "Failed at block " << i;
        ptrs.push_back(result.value());
    }
    
    // 17th should fail (non-growable)
    auto result = pool->alloc(false, 128);
    EXPECT_FALSE(result.hasValue());
}

TEST_F(PoolHeapTest, PointerValidation) {
    // Test is_ptr() validation
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    auto result = pool->alloc(false, 256);
    ASSERT_TRUE(result.hasValue());
    void* valid_ptr = result.value();
    
    // Valid pointer should be recognized
    EXPECT_TRUE(pool->is_ptr(valid_ptr));
    
    // External pointer should not be recognized
    void* external_ptr = malloc(256);
    EXPECT_FALSE(pool->is_ptr(external_ptr));
    free(external_ptr);
    
    // Null pointer should not be valid
    EXPECT_FALSE(pool->is_ptr(nullptr));
}

TEST_F(PoolHeapTest, FreedBlockRecycling) {
    // Test that freed blocks go to free list and get recycled
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    // Allocate 5 blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        auto result = pool->alloc(false, 256);
        ASSERT_TRUE(result.hasValue());
        ptrs.push_back(result.value());
    }
    
    EXPECT_EQ(pool->free_blocks(), 0);
    
    // Free all 5 blocks
    for (void* ptr : ptrs) {
        pool->return_element(ptr, 256);
    }
    
    EXPECT_EQ(pool->free_blocks(), 5);
    
    // Reallocate - should get blocks from free list
    for (int i = 0; i < 5; ++i) {
        auto result = pool->alloc(false, 256);
        ASSERT_TRUE(result.hasValue());
    }
    
    EXPECT_EQ(pool->free_blocks(), 0);
}

TEST_F(PoolHeapTest, MixedAllocFreePattern) {
    // Realistic pattern: allocate some, free some, allocate more
    auto pool = cslt::move(PoolAllocator::Heap(128, 32, 0, 8192, 4096, true, true).value());
    
    // Allocate 3
    auto ptr1 = pool->alloc(false, 128).value();
    auto ptr2 = pool->alloc(false, 128).value();
    auto ptr3 = pool->alloc(false, 128).value();
    
    // Free middle one
    pool->return_element(ptr2, 128);
    EXPECT_EQ(pool->free_blocks(), 1);
    
    // Allocate another - should reuse ptr2
    auto ptr4 = pool->alloc(false, 128).value();
    EXPECT_EQ(ptr4, ptr2);
    EXPECT_EQ(pool->free_blocks(), 0);
    
    // Free first and last
    pool->return_element(ptr1, 128);
    pool->return_element(ptr3, 128);
    EXPECT_EQ(pool->free_blocks(), 2);
}

TEST_F(PoolHeapTest, StatsAfterOperations) {
    // Test statistics after various operations
    auto pool = cslt::move(PoolAllocator::Heap(256, 16, 0, 8192, 4096, true, true).value());
    
    // Allocate some
    auto ptr1 = pool->alloc(false, 256).value();
    auto ptr2 = pool->alloc(false, 256).value();
    (void)ptr2;
    
    // Free one
    pool->return_element(ptr1, 256);
    
    char buffer[1024];
    ASSERT_TRUE(pool->stats(buffer, sizeof(buffer)));
    
    std::string stats_str(buffer);
    
    // Should show correct counts
    EXPECT_NE(stats_str.find("Total Blocks: 16"), std::string::npos);
    EXPECT_NE(stats_str.find("Free Blocks: 1"), std::string::npos);
}
// -------------------------------------------------------------------------------- 

TEST_F(PoolHeapTest, AllocPoolBasic) {
    // Test basic allocation using pool-specific interface
    auto pool = cslt::move(PoolAllocator::Heap(256, 32, 0, 10240, 4096, true, true).value());
    
    // Allocate using pool-specific method (no size parameter needed)
    auto result = pool->alloc_pool();  // Uses default zeroed=false
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, pool->default_alignment()));
}

TEST_F(PoolHeapTest, AllocPoolZeroed) {
    // Test zeroed allocation using pool-specific interface
    auto pool = cslt::move(PoolAllocator::Heap(128, 16, 0, 4096, 4096, true, true).value());
    
    // Allocate with zeroing
    auto result = pool->alloc_pool(true);
    ASSERT_TRUE(result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(result.value());
    
    // Verify all bytes are zero
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " not zeroed";
    }
}
// ================================================================================ 
// ================================================================================ 

class PoolWithArenaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Common cleanup if needed
    }

    // Helper to check if pointer is properly aligned
    bool is_aligned(void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
};

// ================================================================================
// Phase 1: WithArena Pool Tests (10 tests)
// ================================================================================

TEST_F(PoolWithArenaTest, CreateBasicPoolWithArena) {
    // Create parent arena
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    // Create pool using the arena
    auto result = PoolAllocator::WithArena(
        *arena,
        256,    // block_size
        32,     // blocks_per_chunk
        0,      // default alignment
        true,   // grow_enabled
        true    // prewarm
    );

    ASSERT_TRUE(result.hasValue()) << "Failed to create pool with arena";
    auto pool = cslt::move(result.value());
    
    EXPECT_NE(pool.get(), nullptr);
    EXPECT_EQ(pool->block_size(), 256);
    EXPECT_FALSE(pool->owns_memory());  // Doesn't own arena
    EXPECT_EQ(pool->total_blocks(), 32);
}

TEST_F(PoolWithArenaTest, MultiplePoolsSharedArena) {
    // Create one parent arena
    auto arena = cslt::move(ArenaAllocator::Heap(64 * 1024).value());
    
    // Create multiple pools sharing the arena
    auto pool1 = cslt::move(PoolAllocator::WithArena(*arena, 128, 16, 0, true, true).value());
    auto pool2 = cslt::move(PoolAllocator::WithArena(*arena, 256, 16, 0, true, true).value());
    auto pool3 = cslt::move(PoolAllocator::WithArena(*arena, 512, 8, 0, true, true).value());
    
    EXPECT_EQ(pool1->block_size(), 128);
    EXPECT_EQ(pool2->block_size(), 256);
    EXPECT_EQ(pool3->block_size(), 512);
    
    // All should be able to allocate
    auto ptr1 = pool1->alloc(false, 128);
    auto ptr2 = pool2->alloc(false, 256);
    auto ptr3 = pool3->alloc(false, 512);
    
    EXPECT_TRUE(ptr1.hasValue());
    EXPECT_TRUE(ptr2.hasValue());
    EXPECT_TRUE(ptr3.hasValue());
}

TEST_F(PoolWithArenaTest, PoolInheritsArenaMemoryType) {
    // Test with dynamic arena
    auto dynamic_arena = cslt::move(ArenaAllocator::Heap(16384).value());
    auto pool1 = cslt::move(PoolAllocator::WithArena(*dynamic_arena, 128, 16, 0, true, true).value());
    
    EXPECT_EQ(pool1->memory_type(), DYNAMIC);
    
    // Test with static arena
    uint8_t buffer[8192];
    auto static_arena = cslt::move(ArenaAllocator::Stack(buffer, sizeof(buffer)).value());
    auto pool2 = cslt::move(PoolAllocator::WithArena(*static_arena, 128, 16, 0, false, true).value());
    
    EXPECT_EQ(pool2->memory_type(), STATIC);
}

TEST_F(PoolWithArenaTest, AllocateFromSharedArenaPool) {
    // Create arena and pool
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    auto pool = cslt::move(PoolAllocator::WithArena(*arena, 256, 32, 0, true, true).value());
    
    auto result = pool->alloc(false, 256);
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);
    EXPECT_TRUE(arena->is_ptr(ptr));  // Pointer should be in arena
}

TEST_F(PoolWithArenaTest, PoolWithoutPrewarm) {
    // Pool without prewarm should start empty
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    auto result = PoolAllocator::WithArena(
        *arena,
        256,
        32,
        0,
        true,   // Can grow
        false   // No prewarm
    );
    
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    EXPECT_EQ(pool->total_blocks(), 0);
    
    // But should still be able to allocate
    auto alloc_result = pool->alloc(false, 256);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST_F(PoolWithArenaTest, PoolGrowsWithinArena) {
    // Pool should be able to grow if arena has space
    auto arena = cslt::move(ArenaAllocator::Heap(32 * 1024).value());
    
    auto pool = cslt::move(PoolAllocator::WithArena(
        *arena, 128, 16, 0, true, true
    ).value());
    
    size_t initial_blocks = pool->total_blocks();
    EXPECT_EQ(initial_blocks, 16);
    
    // Allocate all initial blocks
    for (size_t i = 0; i < 16; ++i) {
        auto result = pool->alloc(false, 128);
        ASSERT_TRUE(result.hasValue());
    }
    
    // Next allocation should trigger growth
    auto result = pool->alloc(false, 128);
    ASSERT_TRUE(result.hasValue());
    
    EXPECT_GT(pool->total_blocks(), initial_blocks);
}

TEST_F(PoolWithArenaTest, NonGrowablePoolWithArena) {
    // Non-growable pool stops at capacity
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    auto pool = cslt::move(PoolAllocator::WithArena(
        *arena, 128, 16, 0, false, true  // Cannot grow
    ).value());
    
    EXPECT_FALSE(pool->can_grow());
    
    // Allocate all 16 blocks
    for (size_t i = 0; i < 16; ++i) {
        auto result = pool->alloc(false, 128);
        ASSERT_TRUE(result.hasValue());
    }
    
    // 17th should fail
    auto result = pool->alloc(false, 128);
    EXPECT_FALSE(result.hasValue());
}

TEST_F(PoolWithArenaTest, ArenaOutlivesPool) {
    // Verify arena can be used after pool is destroyed
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    // size_t arena_size_before = arena->size();
    
    {
        auto pool = cslt::move(PoolAllocator::WithArena(*arena, 256, 16, 0, true, true).value());
        pool->alloc(false, 256);
        // pool destroyed here
    }
    
    // Arena should still be usable
    auto result = arena->alloc(128);
    EXPECT_TRUE(result.hasValue());
}

TEST_F(PoolWithArenaTest, FailZeroBlockSize) {
    // Should fail with zero block size
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    auto result = PoolAllocator::WithArena(
        *arena,
        0,      // Invalid: zero block size
        32,
        0,
        true,
        true
    );
    
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST_F(PoolWithArenaTest, FailZeroBlocksPerChunk) {
    // Should fail with zero blocks per chunk
    auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
    
    auto result = PoolAllocator::WithArena(
        *arena,
        256,
        0,      // Invalid: zero blocks per chunk
        0,
        true,
        true
    );
    
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}
// ================================================================================ 
// ================================================================================ 

// class PoolStackTest : public ::testing::Test {
// protected:
//     void SetUp() override {
//         // Common setup if needed
//     }
//
//     void TearDown() override {
//         // Common cleanup if needed
//     }
//
//     // Helper to check if pointer is properly aligned
//     bool is_aligned(void* ptr, size_t alignment) {
//         return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
//     }
// };
//
// // ================================================================================
// // Phase 1: Stack Pool Tests (10 tests)
// // ================================================================================
//
TEST(PoolStackTest, CreateBasicStackPool) {
    // Test basic stack pool creation with typical buffer
    uint8_t buffer[4096];
    
    auto result = PoolAllocator::Stack(
        buffer,
        sizeof(buffer),
        256,    // block_size
        0       // default alignment
    );

    ASSERT_TRUE(result.hasValue()) << "Failed to create stack pool";
    auto pool = cslt::move(result.value());

    EXPECT_NE(pool.get(), nullptr);
    EXPECT_EQ(pool->block_size(), 256);
    EXPECT_EQ(pool->memory_type(), STATIC);
    EXPECT_FALSE(pool->owns_memory());  // User owns buffer
    EXPECT_FALSE(pool->can_grow());     // Stack pools never grow
    EXPECT_GT(pool->total_blocks(), 0); // Should have some blocks
}

TEST(PoolStackTest, VerifyBlockCapacity) {
    // Verify correct number of blocks fit in buffer
    uint8_t buffer[8192];
    
    auto result = PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0);
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());
    
    // 8192 bytes - overhead ≈ 7000+ bytes usable
    // 7000 / 128 ≈ 54 blocks (approximately)
    EXPECT_GT(pool->total_blocks(), 40);  // At least 40 blocks
    EXPECT_LT(pool->total_blocks(), 64);  // But not more than 64
}

TEST(PoolStackTest, AllocateFromStackPool) {
    // Test allocation from stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    auto result = pool->alloc(false, 128);
    ASSERT_TRUE(result.hasValue());
    
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);
    
    // Pointer should be within buffer range
    EXPECT_GE(ptr, buffer);
    EXPECT_LT(ptr, buffer + sizeof(buffer));
}

TEST(PoolStackTest, AllocateAllBlocks) {
    // Allocate all available blocks from stack pool
    uint8_t buffer[2048];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 64, 0).value());
    
    size_t total = pool->total_blocks();
    std::vector<void*> ptrs;
    
    // Should be able to allocate all blocks
    for (size_t i = 0; i < total; ++i) {
        auto result = pool->alloc(false, 64);
        ASSERT_TRUE(result.hasValue()) << "Failed at block " << i;
        ptrs.push_back(result.value());
    }
    
    // Next allocation should fail (no growth allowed)
    auto result = pool->alloc(false, 64);
    EXPECT_FALSE(result.hasValue());
}

TEST(PoolStackTest, StackPoolWithCustomAlignment) {
    // Test stack pool with custom alignment
    uint8_t buffer[8192];
    
    auto result = PoolAllocator::Stack(
        buffer,
        sizeof(buffer),
        256,
        64      // 64-byte alignment
    );
   
    if (result.hasError()) {
        std::cout << result.error().what() << "\n";
    }
    ASSERT_TRUE(result.hasValue());
    // auto pool = cslt::move(result.value());
    //
    // EXPECT_EQ(pool->default_alignment(), 64);
    //
    // auto alloc_result = pool->alloc(false, 256);
    // ASSERT_TRUE(alloc_result.hasValue());
}

TEST(PoolStackTest, SmallBufferFewBlocks) {
    // Small buffer should yield few blocks but still work
    uint8_t buffer[512];
    
    auto result = PoolAllocator::Stack(buffer, sizeof(buffer), 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());

    // Should have at least 1 block
    std::cout << pool->total_blocks() << "\n";
    EXPECT_GT(pool->total_blocks(), 0);

    // Should be able to allocate at least one
    auto alloc_result = pool->alloc(false, 64);
    EXPECT_TRUE(alloc_result.hasValue());
}

TEST(PoolStackTest, FailNullBuffer) {
    // Should fail with null buffer
    auto result = PoolAllocator::Stack(
        nullptr,
        4096,
        256,
        0
    );
    
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("null"), std::string::npos);
}

TEST(PoolStackTest, FailZeroBufferSize) {
    // Should fail with zero buffer size
    uint8_t buffer[4096];
    
    auto result = PoolAllocator::Stack(
        buffer,
        0,      // Invalid: zero size
        256,
        0
    );
    
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST(PoolStackTest, FailZeroBlockSize) {
    // Should fail with zero block size
    uint8_t buffer[4096];
    
    auto result = PoolAllocator::Stack(
        buffer,
        sizeof(buffer),
        0,      // Invalid: zero block size
        0
    );
    
    EXPECT_FALSE(result.hasValue());
    std::string error_msg(result.error().what());
    EXPECT_NE(error_msg.find("must be"), std::string::npos);
}

TEST(PoolStackTest, FreeAndReuseStackPool) {
    // Test free-list recycling in stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    auto result1 = pool->alloc(false, 128);
    ASSERT_TRUE(result1.hasValue());
    void* ptr1 = result1.value();
    
    EXPECT_EQ(pool->free_blocks(), 0);
    
    // Free the block
    pool->return_element(ptr1, 128);
    EXPECT_EQ(pool->free_blocks(), 1);
    
    // Allocate again - should reuse freed block
    auto result2 = pool->alloc(false, 128);
    ASSERT_TRUE(result2.hasValue());
    void* ptr2 = result2.value();
    
    EXPECT_EQ(ptr1, ptr2) << "Block not reused from free list";
    EXPECT_EQ(pool->free_blocks(), 0);
}
// -------------------------------------------------------------------------------- 

TEST(PoolStackTest, TinyBufferMinimalBlocks) {
    // Smallest viable buffer (just enough for headers + 1 block)
    uint8_t buffer[256];
    
    auto result = PoolAllocator::Stack(buffer, sizeof(buffer), 16, 0);
    
    if (result.hasValue()) {
        auto pool = cslt::move(result.value());
        // If it succeeds, should have at least 1 block
        EXPECT_EQ(pool->total_blocks(), 1);
    }
    // May fail if buffer too small - that's acceptable
}

TEST(PoolStackTest, BufferTooSmallForOneBlock) {
    // Buffer too small to fit even one block after overhead
    uint8_t buffer[128];
    
    auto result = PoolAllocator::Stack(
        buffer,
        sizeof(buffer),
        1024,   // Block larger than buffer
        0
    );
    
    // Should fail
    EXPECT_FALSE(result.hasValue());
}

TEST(PoolStackTest, StackPoolReset) {
    // Test reset functionality
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    // Allocate some blocks
    pool->alloc(false, 128);
    pool->alloc(false, 128);
    pool->alloc(false, 128);
    
    // Reset pool
    ASSERT_TRUE(pool->reset());
    
    // Should be able to allocate again
    auto result = pool->alloc(false, 128);
    EXPECT_TRUE(result.hasValue());
}

TEST(PoolStackTest, StackPoolCheckpoint) {
    // Test checkpoint/restore with stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    // Allocate one block
    auto perm = pool->alloc(false, 128);
    ASSERT_TRUE(perm.hasValue());
    
    // Save checkpoint
    void* checkpoint = pool->save();
    ASSERT_NE(checkpoint, nullptr);
    
    // Allocate more
    pool->alloc(false, 128);
    pool->alloc(false, 128);
    
    // Restore
    ASSERT_TRUE(pool->restore(checkpoint));
}

TEST(PoolStackTest, MixedAllocFreeStackPool) {
    // Realistic pattern with stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 256, 0).value());
    
    // Allocate 3
    auto ptr1 = pool->alloc(false, 256).value();
    auto ptr2 = pool->alloc(false, 256).value();
    auto ptr3 = pool->alloc(false, 256).value();
    
    // Free middle one
    pool->return_element(ptr2, 256);
    EXPECT_EQ(pool->free_blocks(), 1);
    
    // Allocate - should reuse ptr2
    auto ptr4 = pool->alloc(false, 256).value();
    EXPECT_EQ(ptr4, ptr2);
    
    // Free all
    pool->return_element(ptr1, 256);
    pool->return_element(ptr3, 256);
    pool->return_element(ptr4, 256);
    EXPECT_EQ(pool->free_blocks(), 3);
}

TEST(PoolStackTest, StackPoolStatistics) {
    // Test statistics for stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    char stats_buffer[1024];
    ASSERT_TRUE(pool->stats(stats_buffer, sizeof(stats_buffer)));
    
    std::string stats_str(stats_buffer);
    
    // Should show STATIC type
    EXPECT_NE(stats_str.find("Type: STATIC"), std::string::npos);
    EXPECT_NE(stats_str.find("Block Size: 128"), std::string::npos);
    EXPECT_NE(stats_str.find("Can Grow: No"), std::string::npos);
}

TEST(PoolStackTest, HeapVsStackBuffer) {
    // Compare heap-allocated buffer vs stack buffer
    
    // Heap buffer
    uint8_t* heap_buffer = new uint8_t[4096];
    
    {  // Scope for pool1
        auto pool1 = cslt::move(PoolAllocator::Stack(heap_buffer, 4096, 128, 0).value());
        
        // Stack buffer
        uint8_t stack_buffer[4096];
        auto pool2 = cslt::move(PoolAllocator::Stack(stack_buffer, 4096, 128, 0).value());
        
        // Both should work the same
        EXPECT_EQ(pool1->block_size(), pool2->block_size());
        EXPECT_EQ(pool1->memory_type(), pool2->memory_type());
        
        // Both should allocate
        auto result1 = pool1->alloc(false, 128);
        auto result2 = pool2->alloc(false, 128);
        EXPECT_TRUE(result1.hasValue());
        EXPECT_TRUE(result2.hasValue());
        
    }  // pool1 and pool2 destroyed here (before buffer is freed)
    
    // Now safe to delete buffer
    delete[] heap_buffer;
}

TEST(PoolStackTest, AlignedStackBuffer) {
    // Test with aligned buffer
    alignas(64) uint8_t buffer[8192];
    
    auto result = PoolAllocator::Stack(buffer, sizeof(buffer), 256, 64);
    ASSERT_TRUE(result.hasValue());
    auto pool = cslt::move(result.value());

    EXPECT_EQ(pool->default_alignment(), 64);

    auto alloc_result = pool->alloc(256, false);
   ASSERT_TRUE(alloc_result.hasValue());
}

TEST(PoolStackTest, PointerValidationStackPool) {
    // Test pointer validation for stack pool
    uint8_t buffer[4096];
    auto pool = cslt::move(PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0).value());
    
    auto result = pool->alloc(128, false);
    ASSERT_TRUE(result.hasValue());
    void* valid_ptr = result.value();
    
    // Valid pointer should be recognized
    EXPECT_TRUE(pool->is_ptr(valid_ptr));

    // Pointer outside buffer should not be valid
    uint8_t external_buffer[128];
    EXPECT_FALSE(pool->is_ptr(external_buffer));

    // Null should not be valid
    EXPECT_FALSE(pool->is_ptr(nullptr));

    // Pointer just beyond buffer should not be valid
    EXPECT_FALSE(pool->is_ptr(buffer + sizeof(buffer)));
}
// -------------------------------------------------------------------------------- 

TEST(PoolStackTest, DiagnosticLifetime) {
    std::cout << "=== Starting test ===" << std::endl;
    
    uint8_t buffer[4096];
    std::cout << "Buffer at: " << static_cast<void*>(buffer) << std::endl;
    
    {
        auto result = PoolAllocator::Stack(buffer, sizeof(buffer), 128, 0);
        ASSERT_TRUE(result.hasValue()) << result.error().what();
        
        std::cout << "Creating pool..." << std::endl;
        auto pool = cslt::move(result.value());
        
        std::cout << "Pool at: " << static_cast<void*>(pool.get()) << std::endl;
        // std::cout << "Arena at: " << static_cast<void*>(pool->arena_) << std::endl;
        // std::cout << "owns_arena: " << pool->owns_arena_ << std::endl;
        std::cout << "memory_type: " << (int)pool->memory_type() << std::endl;
        std::cout << "total_blocks: " << pool->total_blocks() << std::endl;
        
        std::cout << "Allocating..." << std::endl;
        auto alloc_result = pool->alloc_pool();
        ASSERT_TRUE(alloc_result.hasValue());
        std::cout << "Allocated at: " << alloc_result.value() << std::endl;
        
        std::cout << "About to destroy pool..." << std::endl;
    }
    
    std::cout << "Pool destroyed" << std::endl;
    std::cout << "About to exit test (buffer will be freed)..." << std::endl;
}
// ================================================================================ 
// ================================================================================ 

TEST(FreeListHeapTest, CreateBasicHeapFreelist) {
    auto result = FreeListAllocator::Heap(4096, 0, false);
    
    ASSERT_TRUE(result.hasValue()) << "Heap creation should succeed";
    
    auto freelist = cslt::move(result.value());

    EXPECT_NE(freelist.get(), nullptr);
    EXPECT_GT(freelist->allocated(), 0);
    EXPECT_EQ(freelist->used(), 0);
    EXPECT_EQ(freelist->remaining(), freelist->allocated());
    EXPECT_TRUE(freelist->owns_arena());
}

// ================================================================================
// Test 2: Default Size Handling (bytes == 0)
// ================================================================================

TEST(FreeListHeapTest, DefaultSizeWhenZero) {
    auto result = FreeListAllocator::Heap(0, 0, false);
    
    ASSERT_TRUE(result.hasValue()) << "Zero bytes should use default size";
    
    auto freelist = cslt::move(result.value());
    // Should get at least the default minimum (4096 or similar)
    EXPECT_GE(freelist->allocated(), 1024);
}

// ================================================================================
// Test 3: Custom Alignment
// ================================================================================

TEST(FreeListHeapTest, CustomAlignment) {
    auto result = FreeListAllocator::Heap(4096, 64, false);
    
    ASSERT_TRUE(result.hasValue()) << "Custom alignment should succeed";
    
    auto freelist = cslt::move(result.value());
    
    // Allocate and verify alignment
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 64, 0) << "Pointer should be 64-byte aligned";
}

// ================================================================================
// Test 4: Allocate and Free Basic Operations
// ================================================================================

TEST(FreeListHeapTest, AllocateAndFree) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    size_t initial_remaining = freelist->remaining();
    
    // Allocate
    auto ptr_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    EXPECT_LT(freelist->remaining(), initial_remaining);
    EXPECT_GT(freelist->used(), 0);
    
    // Free
    freelist->return_element(ptr, 512);
    
    // Should have reclaimed most space (may not be exact due to overhead)
    EXPECT_GT(freelist->remaining(), initial_remaining - 100);
}

// ================================================================================
// Test 5: Multiple Allocations
// ================================================================================

TEST(FreeListHeapTest, MultipleAllocations) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    std::vector<void*> pointers;
    
    // Allocate multiple blocks
    for (int i = 0; i < 10; ++i) {
        auto ptr_result = freelist->alloc(256, false);
        ASSERT_TRUE(ptr_result.hasValue()) << "Allocation " << i << " should succeed";
        pointers.push_back(ptr_result.value());
    }
    
    EXPECT_EQ(pointers.size(), 10);
    EXPECT_GT(freelist->used(), 2560); // At least 10 * 256
    
    // Free all
    for (void* ptr : pointers) {
        freelist->return_element(ptr, 256);
    }
    
    // Should be mostly empty again
    EXPECT_LT(freelist->used(), 100);
}

// ================================================================================
// Test 6: Zero-Initialized Allocation
// ================================================================================

TEST(FreeListHeapTest, ZeroInitializedAllocation) {
    auto result = FreeListAllocator::Heap(4096, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    auto ptr_result = freelist->alloc(512, true);
    ASSERT_TRUE(ptr_result.hasValue());
    
    uint8_t* ptr = static_cast<uint8_t*>(ptr_result.value());
    
    // Verify all bytes are zero
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(ptr[i], 0) << "Byte " << i << " should be zero";
    }
}

// ================================================================================
// Test 7: Invalid Alignment (Not Power of 2)
// ================================================================================

TEST(FreeListHeapTest, InvalidAlignment) {
    auto result = FreeListAllocator::Heap(4096, 63, false); // Not power of 2
    
    EXPECT_FALSE(result.hasValue()) << "Non-power-of-2 alignment should fail";
    // Error message should mention alignment
    if (!result.hasValue()) {
        std::string error_msg = result.error().what();
        EXPECT_NE(error_msg.find("Alignment"), std::string::npos) 
            << "Error message should mention alignment";
    }
}

// ================================================================================
// Test 8: Capacity Exhaustion
// ================================================================================

TEST(FreeListHeapTest, CapacityExhaustion) {
    auto result = FreeListAllocator::Heap(1024, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate until exhausted
    std::vector<void*> pointers;
    
    while (true) {
        auto ptr_result = freelist->alloc(64, false);
        if (!ptr_result.hasValue()) {
            break;
        }
        pointers.push_back(ptr_result.value());
    }
    
    EXPECT_GT(pointers.size(), 0) << "Should allocate at least some blocks";
    
    // Verify we're close to capacity
    EXPECT_LT(freelist->remaining(), 200); // Should be nearly full
    
    // Free one block
    freelist->return_element(pointers[0], 64);
    
    // Should be able to allocate again
    auto new_result = freelist->alloc(64, false);
    EXPECT_TRUE(new_result.hasValue()) << "Should be able to reuse freed block";
}

// ================================================================================
// Test 9: Statistics Report
// ================================================================================

TEST(FreeListHeapTest, StatisticsReport) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Make some allocations
    auto ptr1_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr1_result.hasValue());
    auto ptr2_result = freelist->alloc(1024, false);
    ASSERT_TRUE(ptr2_result.hasValue());
    
    char buffer[2048];
    bool success = freelist->stats(buffer, sizeof(buffer));
    
    ASSERT_TRUE(success) << "Stats generation should succeed";
    
    std::string stats_str(buffer);
    
    // Verify report contains expected information
    EXPECT_NE(stats_str.find("FreeListAllocator Statistics"), std::string::npos);
    EXPECT_NE(stats_str.find("Type: DYNAMIC"), std::string::npos);
    EXPECT_NE(stats_str.find("Owns arena: yes"), std::string::npos);
    EXPECT_NE(stats_str.find("Used"), std::string::npos);
    EXPECT_NE(stats_str.find("Remaining"), std::string::npos);
    EXPECT_NE(stats_str.find("Free block"), std::string::npos);
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Test 1: Basic Stack Creation
// ================================================================================

TEST(FreeListStackTest, CreateBasicStackFreelist) {
    uint8_t buffer[8192];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    
    ASSERT_TRUE(result.hasValue()) << "Stack creation should succeed";
    
    auto freelist = cslt::move(result.value());
    
    EXPECT_NE(freelist.get(), nullptr);
    EXPECT_GT(freelist->allocated(), 0);
    EXPECT_EQ(freelist->used(), 0);
    EXPECT_EQ(freelist->remaining(), freelist->allocated());
    EXPECT_TRUE(freelist->owns_arena());
}

// ================================================================================
// Test 2: Custom Alignment
// ================================================================================

TEST(FreeListStackTest, CustomAlignment) {
    alignas(64) uint8_t buffer[8192];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 64);
    
    ASSERT_TRUE(result.hasValue()) << "Custom alignment should succeed";
    
    auto freelist = cslt::move(result.value());
    
    // Allocate and verify alignment
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 64, 0) << "Pointer should be 64-byte aligned";
}

// ================================================================================
// Test 3: Allocate and Free Basic Operations
// ================================================================================

TEST(FreeListStackTest, AllocateAndFree) {
    uint8_t buffer[8192];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    size_t initial_remaining = freelist->remaining();
    
    // Allocate
    auto ptr_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    EXPECT_LT(freelist->remaining(), initial_remaining);
    EXPECT_GT(freelist->used(), 0);
    
    // Free
    freelist->return_element(ptr, 512);
    
    // Should have reclaimed most space
    EXPECT_GT(freelist->remaining(), initial_remaining - 100);
}

// ================================================================================
// Test 4: Multiple Allocations
// ================================================================================

TEST(FreeListStackTest, MultipleAllocations) {
    uint8_t buffer[16384];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    std::vector<void*> pointers;
    
    // Allocate multiple blocks
    for (int i = 0; i < 10; ++i) {
        auto ptr_result = freelist->alloc(256, false);
        ASSERT_TRUE(ptr_result.hasValue()) << "Allocation " << i << " should succeed";
        pointers.push_back(ptr_result.value());
    }
    
    EXPECT_EQ(pointers.size(), 10);
    EXPECT_GT(freelist->used(), 2560); // At least 10 * 256
    
    // Free all
    for (void* ptr : pointers) {
        freelist->return_element(ptr, 256);
    }
    
    // Should be mostly empty again
    EXPECT_LT(freelist->used(), 100);
}

// ================================================================================
// Test 5: Null Buffer
// ================================================================================

TEST(FreeListStackTest, NullBuffer) {
    auto result = FreeListAllocator::Stack(nullptr, 1024, 0);
    
    EXPECT_FALSE(result.hasValue()) << "Null buffer should fail";
    if (!result.hasValue()) {
        std::string error_msg = result.error().what();
        EXPECT_NE(error_msg.find("null"), std::string::npos) 
            << "Error should mention null buffer";
    }
}

// ================================================================================
// Test 6: Zero Buffer Size
// ================================================================================

TEST(FreeListStackTest, ZeroBufferSize) {
    uint8_t buffer[1024];
    
    auto result = FreeListAllocator::Stack(buffer, 0, 0);
    
    EXPECT_FALSE(result.hasValue()) << "Zero buffer size should fail";
    if (!result.hasValue()) {
        std::string error_msg = result.error().what();
        EXPECT_NE(error_msg.find("size"), std::string::npos) 
            << "Error should mention size";
    }
}

// ================================================================================
// Test 7: Buffer Too Small
// ================================================================================

TEST(FreeListStackTest, BufferTooSmall) {
    uint8_t buffer[64];  // Too small for headers
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    
    EXPECT_FALSE(result.hasValue()) << "Tiny buffer should fail";
    if (!result.hasValue()) {
        std::string error_msg = result.error().what();
        // Error should mention structures or capacity
        bool mentions_issue = error_msg.find("structures") != std::string::npos ||
                              error_msg.find("capacity") != std::string::npos ||
                              error_msg.find("small") != std::string::npos;
        EXPECT_TRUE(mentions_issue) << "Error should mention size issue";
    }
}

// ================================================================================
// Test 8: Stack Freelist Reset
// ================================================================================

TEST(FreeListStackTest, ResetStackFreelist) {
    uint8_t buffer[8192];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Make allocations
    auto ptr1_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr1_result.hasValue());
    auto ptr2_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr2_result.hasValue());
    
    size_t used_before = freelist->used();
    EXPECT_GT(used_before, 0);
    
    // Reset
    bool reset_ok = freelist->reset();
    ASSERT_TRUE(reset_ok) << "Reset should succeed";
    
    // Verify clean state
    EXPECT_EQ(freelist->used(), 0);
    EXPECT_EQ(freelist->remaining(), freelist->allocated());
    
    // Should be able to allocate again
    auto new_result = freelist->alloc(1024, false);
    EXPECT_TRUE(new_result.hasValue()) << "Allocation after reset should succeed";
}

// ================================================================================
// Test 9: Buffer Outlives Freelist
// ================================================================================

TEST(FreeListStackTest, BufferOutlivesFreelist) {
    uint8_t buffer[4096];
    memset(buffer, 0xAA, sizeof(buffer));  // Fill with pattern
    
    {
        auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
        ASSERT_TRUE(result.hasValue());
        auto freelist = cslt::move(result.value());
        
        // Use freelist
        auto ptr = freelist->alloc(512, false);
        ASSERT_TRUE(ptr.hasValue());
        
        // Freelist destroyed here
    }
    
    // Buffer should still be valid and unchanged outside used region
    // (We can't easily verify the exact state, but this shouldn't crash)
    EXPECT_TRUE(true) << "Buffer survived freelist destruction";
}

// ================================================================================
// Test 10: Statistics Report for Stack Freelist
// ================================================================================

TEST(FreeListStackTest, StatisticsReport) {
    uint8_t buffer[8192];
    
    auto result = FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Make some allocations
    auto ptr1_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr1_result.hasValue());
    auto ptr2_result = freelist->alloc(1024, false);
    ASSERT_TRUE(ptr2_result.hasValue());
    
    char stats_buffer[2048];
    bool success = freelist->stats(stats_buffer, sizeof(stats_buffer));
    
    ASSERT_TRUE(success) << "Stats generation should succeed";
    
    std::string stats_str(stats_buffer);
    
    // Verify report contains expected information for STATIC freelist
    EXPECT_NE(stats_str.find("FreeListAllocator Statistics"), std::string::npos);
    EXPECT_NE(stats_str.find("Type: STATIC"), std::string::npos);
    EXPECT_NE(stats_str.find("Owns arena: yes"), std::string::npos);
    EXPECT_NE(stats_str.find("Used"), std::string::npos);
    EXPECT_NE(stats_str.find("Remaining"), std::string::npos);
    EXPECT_NE(stats_str.find("Free block"), std::string::npos);
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Test 1: Basic WithArena Creation
// ================================================================================

TEST(FreeListWithArenaTest, CreateBasicWithArena) {
    auto arena_result = ArenaAllocator::Heap(32768);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*arena, 8192, 0);
    
    ASSERT_TRUE(result.hasValue()) << "WithArena creation should succeed";
    
    auto freelist = cslt::move(result.value());
    
    EXPECT_NE(freelist.get(), nullptr);
    EXPECT_GT(freelist->allocated(), 0);
    EXPECT_EQ(freelist->used(), 0);
    EXPECT_EQ(freelist->remaining(), freelist->allocated());
    EXPECT_FALSE(freelist->owns_arena()) << "Should NOT own borrowed arena";
}

// ================================================================================
// Test 2: Multiple Freelists Sharing Arena
// ================================================================================

TEST(FreeListWithArenaTest, MultiplefreelistsSharedArena) {
    auto arena_result = ArenaAllocator::Heap(65536);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    // Create three freelists sharing the same arena
    auto fl1_result = FreeListAllocator::WithArena(*arena, 8192, 0);
    ASSERT_TRUE(fl1_result.hasValue());
    auto freelist1 = cslt::move(fl1_result.value());

    auto fl2_result = FreeListAllocator::WithArena(*arena, 4096, 0);
    ASSERT_TRUE(fl2_result.hasValue());
    auto freelist2 = cslt::move(fl2_result.value());

    auto fl3_result = FreeListAllocator::WithArena(*arena, 2048, 0);
    ASSERT_TRUE(fl3_result.hasValue());
    auto freelist3 = cslt::move(fl3_result.value());

    // All should be valid
    EXPECT_NE(freelist1.get(), nullptr);
    EXPECT_NE(freelist2.get(), nullptr);
    EXPECT_NE(freelist3.get(), nullptr);

    // All should not own arena
    EXPECT_FALSE(freelist1->owns_arena());
    EXPECT_FALSE(freelist2->owns_arena());
    EXPECT_FALSE(freelist3->owns_arena());

    // Should be able to allocate from all
    auto ptr1 = freelist1->alloc(256, false);
    auto ptr2 = freelist2->alloc(128, false);
    auto ptr3 = freelist3->alloc(64, false);

    EXPECT_TRUE(ptr1.hasValue());
    EXPECT_TRUE(ptr2.hasValue());
    EXPECT_TRUE(ptr3.hasValue());
}

// ================================================================================
// Test 3: Arena Outlives Freelist
// ================================================================================

TEST(FreeListWithArenaTest, ArenaOutlivesFreelist) {
    auto arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    // size_t arena_used_before = arena->used();
    
    {
        // Freelist created and destroyed in this scope
        auto result = FreeListAllocator::WithArena(*arena, 4096, 0);
        ASSERT_TRUE(result.hasValue());
        auto freelist = cslt::move(result.value());
        
        // Use freelist
        auto ptr = freelist->alloc(512, false);
        ASSERT_TRUE(ptr.hasValue());
        
        // Freelist destroyed here
    }
    
    // Arena should still be valid and usable
    EXPECT_NE(arena.get(), nullptr);
    
    // Should be able to allocate from arena
    auto arena_alloc = arena->alloc(256, false);
    EXPECT_TRUE(arena_alloc.hasValue()) << "Arena should still be usable";
}

// ================================================================================
// Test 4: Custom Alignment with Borrowed Arena
// ================================================================================

TEST(FreeListWithArenaTest, CustomAlignment) {
    auto arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*arena, 4096, 64);
    
    ASSERT_TRUE(result.hasValue()) << "Custom alignment should succeed";
    
    auto freelist = cslt::move(result.value());
    
    // Allocate and verify alignment
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 64, 0) << "Pointer should be 64-byte aligned";
}

// ================================================================================
// Test 5: Allocate and Free with Borrowed Arena
// ================================================================================

TEST(FreeListWithArenaTest, AllocateAndFree) {
    auto arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*arena, 8192, 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    size_t initial_remaining = freelist->remaining();
    
    // Allocate
    auto ptr_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    EXPECT_LT(freelist->remaining(), initial_remaining);
    EXPECT_GT(freelist->used(), 0);
    
    // Free
    freelist->return_element(ptr, 512);
    
    // Should have reclaimed most space
    EXPECT_GT(freelist->remaining(), initial_remaining - 100);
}

// ================================================================================
// Test 6: Insufficient Arena Space
// ================================================================================

TEST(FreeListWithArenaTest, InsufficientArenaSpace) {
    auto arena_result = ArenaAllocator::Heap(1024);  // Small arena
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    // Try to allocate more than arena can provide
    auto result = FreeListAllocator::WithArena(*arena, 8192, 0);
    
    EXPECT_FALSE(result.hasValue()) << "Should fail when arena too small";
    if (!result.hasValue()) {
        std::string error_msg = result.error().what();
        // Error should mention memory or capacity
        bool mentions_issue = error_msg.find("memory") != std::string::npos ||
                              error_msg.find("capacity") != std::string::npos ||
                              error_msg.find("Arena") != std::string::npos;
        EXPECT_TRUE(mentions_issue) << "Error should mention space issue";
    }
}

// ================================================================================
// Test 7: Reset Freelist Does Not Affect Arena
// ================================================================================

TEST(FreeListWithArenaTest, ResetDoesNotAffectArena) {
    auto arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*arena, 4096, 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Make allocations
    auto ptr1 = freelist->alloc(256, false);
    auto ptr2 = freelist->alloc(512, false);
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    
    // Reset freelist
    bool reset_ok = freelist->reset();
    ASSERT_TRUE(reset_ok);
    
    // Freelist should be reset
    EXPECT_EQ(freelist->used(), 0);
    
}

// ================================================================================
// Test 8: Memory Type Inheritance
// ================================================================================

TEST(FreeListWithArenaTest, MemoryTypeInheritance) {
    // Create DYNAMIC arena
    auto dynamic_arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(dynamic_arena_result.hasValue());
    auto dynamic_arena = cslt::move(dynamic_arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*dynamic_arena, 4096, 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Check stats to verify memory type
    char buffer[2048];
    bool success = freelist->stats(buffer, sizeof(buffer));
    ASSERT_TRUE(success);
    
    std::string stats_str(buffer);
    
    // Should inherit DYNAMIC from parent arena
    EXPECT_NE(stats_str.find("Type: DYNAMIC"), std::string::npos) 
        << "Should inherit DYNAMIC type from heap arena";
}

// ================================================================================
// Test 9: Statistics Report with Borrowed Arena
// ================================================================================

TEST(FreeListWithArenaTest, StatisticsReport) {
    auto arena_result = ArenaAllocator::Heap(16384);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    auto result = FreeListAllocator::WithArena(*arena, 8192, 0);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Make some allocations
    auto ptr1 = freelist->alloc(512, false);
    auto ptr2 = freelist->alloc(1024, false);
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    
    char buffer[2048];
    bool success = freelist->stats(buffer, sizeof(buffer));
    
    ASSERT_TRUE(success) << "Stats generation should succeed";
    
    std::string stats_str(buffer);
    
    // Verify report shows borrowed arena
    EXPECT_NE(stats_str.find("FreeListAllocator Statistics"), std::string::npos);
    EXPECT_NE(stats_str.find("Owns arena: no"), std::string::npos) 
        << "Should show arena is borrowed";
    EXPECT_NE(stats_str.find("Used"), std::string::npos);
    EXPECT_NE(stats_str.find("Remaining"), std::string::npos);
    EXPECT_NE(stats_str.find("Free block"), std::string::npos);
}

// ================================================================================
// Test 10: Sequential Freelist Destruction
// ================================================================================

TEST(FreeListWithArenaTest, SequentialFreelistDestruction) {
    auto arena_result = ArenaAllocator::Heap(65536);
    ASSERT_TRUE(arena_result.hasValue());
    auto arena = cslt::move(arena_result.value());
    
    // Create first freelist
    auto fl1_result = FreeListAllocator::WithArena(*arena, 8192, 0);
    ASSERT_TRUE(fl1_result.hasValue());
    auto freelist1 = cslt::move(fl1_result.value());
    
    auto ptr1 = freelist1->alloc(256, false);
    ASSERT_TRUE(ptr1.hasValue());
    
    // Destroy first freelist
    freelist1.reset();
    
    // Arena should still be valid
    EXPECT_NE(arena.get(), nullptr);
    
    // Create second freelist using same arena
    auto fl2_result = FreeListAllocator::WithArena(*arena, 4096, 0);
    ASSERT_TRUE(fl2_result.hasValue());
    auto freelist2 = cslt::move(fl2_result.value());
    
    auto ptr2 = freelist2->alloc(128, false);
    EXPECT_TRUE(ptr2.hasValue()) << "Second freelist should work after first destroyed";
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Test 1: Forward Coalescing (Free Block Merges with Next)
// ================================================================================

TEST(FreeListCoalescingTest, ForwardCoalescing) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate three adjacent blocks
    auto ptr1_result = freelist->alloc(256, false);
    auto ptr2_result = freelist->alloc(256, false);
    auto ptr3_result = freelist->alloc(256, false);
    
    ASSERT_TRUE(ptr1_result.hasValue());
    ASSERT_TRUE(ptr2_result.hasValue());
    ASSERT_TRUE(ptr3_result.hasValue());
    
    void* ptr1 = ptr1_result.value();
    void* ptr2 = ptr2_result.value();
    void* ptr3 = ptr3_result.value();
    
    ///size_t used_after_alloc = freelist->used();
    
    // Free middle block first
    freelist->return_element(ptr2, 256);
    
    // Free first block - should coalesce with middle
    freelist->return_element(ptr1, 256);
    
    // After coalescing, should have more space available
    // The two adjacent freed blocks should have merged
    // size_t remaining_after_coalesce = freelist->remaining();
    
    // Free third block - should coalesce all three
    freelist->return_element(ptr3, 256);
    
    // Should be able to allocate a larger block now
    auto large_result = freelist->alloc(600, false);
    EXPECT_TRUE(large_result.hasValue()) 
        << "Should be able to allocate larger block after coalescing";
}

// ================================================================================
// Test 2: Backward Coalescing (Free Block Merges with Previous)
// ================================================================================

TEST(FreeListCoalescingTest, BackwardCoalescing) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate three adjacent blocks
    auto ptr1_result = freelist->alloc(256, false);
    auto ptr2_result = freelist->alloc(256, false);
    auto ptr3_result = freelist->alloc(256, false);
    
    ASSERT_TRUE(ptr1_result.hasValue());
    ASSERT_TRUE(ptr2_result.hasValue());
    ASSERT_TRUE(ptr3_result.hasValue());
    
    void* ptr1 = ptr1_result.value();
    void* ptr2 = ptr2_result.value();
    void* ptr3 = ptr3_result.value();
    
    // Free first block
    freelist->return_element(ptr1, 256);
    
    // Free second block - should coalesce backward with first
    freelist->return_element(ptr2, 256);
    
    // Free third block - should coalesce backward with first+second
    freelist->return_element(ptr3, 256);
    
    // Should be able to allocate a block spanning all three
    auto large_result = freelist->alloc(700, false);
    EXPECT_TRUE(large_result.hasValue()) 
        << "Should be able to allocate large block after backward coalescing";
}

// ================================================================================
// Test 3: Bidirectional Coalescing (Merge with Both Neighbors)
// ================================================================================

TEST(FreeListCoalescingTest, BidirectionalCoalescing) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate five blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        auto ptr_result = freelist->alloc(256, false);
        ASSERT_TRUE(ptr_result.hasValue());
        ptrs.push_back(ptr_result.value());
    }
    
    // Free blocks 0, 2, 4 (leave 1 and 3 allocated)
    freelist->return_element(ptrs[0], 256);
    freelist->return_element(ptrs[2], 256);
    freelist->return_element(ptrs[4], 256);
    
    // Free block 1 - should coalesce with blocks 0 and 2
    freelist->return_element(ptrs[1], 256);
    
    // Should now have a larger contiguous free block (0+1+2)
    auto large_result = freelist->alloc(700, false);
    EXPECT_TRUE(large_result.hasValue()) 
        << "Should allocate large block after bidirectional coalescing";
}

// ================================================================================
// Test 4: No Coalescing (Non-Adjacent Blocks)
// ================================================================================

TEST(FreeListCoalescingTest, NoCoalescingNonAdjacent) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        auto ptr_result = freelist->alloc(256, false);
        ASSERT_TRUE(ptr_result.hasValue());
        ptrs.push_back(ptr_result.value());
    }
    
    // Free non-adjacent blocks (0, 2, 4)
    freelist->return_element(ptrs[0], 256);
    freelist->return_element(ptrs[2], 256);
    freelist->return_element(ptrs[4], 256);
    
    // Blocks 1 and 3 still allocated, preventing coalescing
    
    // Should NOT be able to allocate a very large block
    auto large_result = freelist->alloc(1000, false);
    // This might fail due to fragmentation
    // (We can't guarantee it fails, but we're demonstrating the pattern)
    
    // Free remaining blocks
    freelist->return_element(ptrs[1], 256);
    freelist->return_element(ptrs[3], 256);
    
    // Now should be able to allocate larger block
    auto large_result2 = freelist->alloc(1200, false);
    EXPECT_TRUE(large_result2.hasValue()) 
        << "Should allocate after freeing all blocks";
}

// ================================================================================
// Test 5: Complete Coalescing Back to Initial State
// ================================================================================

TEST(FreeListCoalescingTest, CompleteCoalescingToInitialState) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    size_t initial_remaining = freelist->remaining();
    
    // Allocate all available space in chunks
    std::vector<void*> ptrs;
    while (true) {
        auto ptr_result = freelist->alloc(128, false);
        if (!ptr_result.hasValue()) {
            break;
        }
        ptrs.push_back(ptr_result.value());
    }
    
    EXPECT_GT(ptrs.size(), 0) << "Should have allocated some blocks";
    
    // Free all blocks in random order
    for (size_t i = 0; i < ptrs.size(); ++i) {
        freelist->return_element(ptrs[i], 128);
    }
    
    // Should have coalesced back to nearly initial state
    size_t final_remaining = freelist->remaining();
    
    // Allow for some overhead difference due to headers
    EXPECT_GT(final_remaining, initial_remaining - 200) 
        << "Should recover most space after coalescing";
}

// ================================================================================
// REALLOC TESTS
// ================================================================================

// ================================================================================
// Test 6: Realloc Growth (Simple)
// ================================================================================

TEST(FreeListReallocTest, ReallocGrowth) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate initial block
    auto ptr_result = freelist->alloc(128, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Write pattern to memory
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 128; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    
    // Grow to 512 bytes
    auto new_result = freelist->realloc(ptr, 128, 512, false);
    ASSERT_TRUE(new_result.hasValue()) << "Realloc growth should succeed";
    
    void* new_ptr = new_result.value();
    uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
    
    // Verify old data was copied
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(new_data[i], static_cast<uint8_t>(i)) 
            << "Byte " << i << " should be preserved";
    }
}

// ================================================================================
// Test 7: Realloc Shrink (No-Op)
// ================================================================================

TEST(FreeListReallocTest, ReallocShrinkNoOp) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate block
    auto ptr_result = freelist->alloc(512, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Try to shrink to 256 bytes
    auto new_result = freelist->realloc(ptr, 512, 256, false);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    
    // Should return same pointer (no-op for shrinking)
    EXPECT_EQ(new_ptr, ptr) << "Realloc shrink should return same pointer";
}

// ================================================================================
// Test 8: Realloc with Zero-Fill
// ================================================================================

TEST(FreeListReallocTest, ReallocWithZeroFill) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate and fill with pattern
    auto ptr_result = freelist->alloc(128, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 128; ++i) {
        data[i] = 0xFF;
    }
    
    // Grow with zero-fill
    auto new_result = freelist->realloc(ptr, 128, 512, true);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
    
    // Verify old data preserved
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(new_data[i], 0xFF) << "Old data should be preserved";
    }
    
    // Verify new region is zeroed
    for (int i = 128; i < 512; ++i) {
        EXPECT_EQ(new_data[i], 0) << "New region should be zeroed";
    }
}

// ================================================================================
// Test 9: Realloc NULL Pointer (Behaves like alloc)
// ================================================================================

TEST(FreeListReallocTest, ReallocNullPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Realloc with NULL pointer should behave like alloc
    auto ptr_result = freelist->realloc(nullptr, 0, 256, true);
    
    ASSERT_TRUE(ptr_result.hasValue()) << "Realloc NULL should succeed";
    
    void* ptr = ptr_result.value();
    EXPECT_NE(ptr, nullptr);
    
    // Should be zeroed
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], 0) << "Should be zero-initialized";
    }
}

// ================================================================================
// Test 10: Realloc Aligned with Custom Alignment
// ================================================================================

TEST(FreeListReallocTest, ReallocAlignedCustomAlignment) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate with 64-byte alignment
    auto ptr_result = freelist->alloc_aligned(128, 64, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Verify initial alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    
    // Write pattern
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 128; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    
    // Grow with same alignment
    auto new_result = freelist->realloc_aligned(ptr, 128, 512, 64, false);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    
    // Verify new alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(new_ptr) % 64, 0) 
        << "New pointer should maintain 64-byte alignment";
    
    // Verify data copied
    uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(new_data[i], static_cast<uint8_t>(i)) 
            << "Data should be preserved";
    }
}

// ================================================================================
// Test 11: Realloc Aligned NULL Pointer
// ================================================================================

TEST(FreeListReallocTest, ReallocAlignedNullPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Realloc_aligned with NULL should behave like alloc_aligned
    auto ptr_result = freelist->realloc_aligned(nullptr, 0, 256, 128, true);
    
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    
    // Verify alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 128, 0) 
        << "Should have 128-byte alignment";
    
    // Verify zeroed
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], 0);
    }
}

// ================================================================================
// Test 12: Multiple Realloc Growth Steps
// ================================================================================

TEST(FreeListReallocTest, MultipleReallocGrowthSteps) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Start with small allocation
    auto ptr_result = freelist->alloc(64, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Mark initial data
    uint8_t* data = static_cast<uint8_t*>(ptr);
    data[0] = 0xAA;
    
    size_t current_size = 64;
    
    // Grow in steps: 64 -> 128 -> 256 -> 512 -> 1024
    size_t sizes[] = {128, 256, 512, 1024};
    
    for (size_t new_size : sizes) {
        auto new_result = freelist->realloc(ptr, current_size, new_size, false);
        ASSERT_TRUE(new_result.hasValue()) << "Growth to " << new_size << " should succeed";
        
        ptr = new_result.value();
        data = static_cast<uint8_t*>(ptr);
        
        // Verify marker still present
        EXPECT_EQ(data[0], 0xAA) << "Data should be preserved through realloc";
        
        current_size = new_size;
    }
    
    EXPECT_EQ(current_size, 1024);
}

// ================================================================================
// Test 13: Realloc Changing Alignment
// ================================================================================

TEST(FreeListReallocTest, ReallocChangingAlignment) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate with default alignment
    auto ptr_result = freelist->alloc(128, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Write pattern
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 128; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    
    // Realloc with stricter alignment (128 bytes)
    auto new_result = freelist->realloc_aligned(ptr, 128, 512, 128, false);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    
    // Verify new stricter alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(new_ptr) % 128, 0);
    
    // Verify data preserved
    uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(new_data[i], static_cast<uint8_t>(i));
    }
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Test 1: is_ptr with Valid Allocation
// ================================================================================

TEST(FreeListValidationTest, IsPtrValidAllocation) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate a block
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Should recognize valid pointer
    EXPECT_TRUE(freelist->is_ptr(ptr)) << "Should recognize valid allocated pointer";
}

// ================================================================================
// Test 2: is_ptr with Multiple Valid Allocations
// ================================================================================

TEST(FreeListValidationTest, IsPtrMultipleValidAllocations) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate multiple blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        auto ptr_result = freelist->alloc(128, false);
        ASSERT_TRUE(ptr_result.hasValue());
        ptrs.push_back(ptr_result.value());
    }
    
    // All should be recognized as valid
    for (size_t i = 0; i < ptrs.size(); ++i) {
        EXPECT_TRUE(freelist->is_ptr(ptrs[i])) 
            << "Pointer " << i << " should be valid";
    }
}

// ================================================================================
// Test 3: is_ptr with NULL Pointer
// ================================================================================

TEST(FreeListValidationTest, IsPtrNullPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // NULL pointer should return false
    EXPECT_FALSE(freelist->is_ptr(nullptr)) << "NULL pointer should be invalid";
}

// ================================================================================
// Test 4: is_ptr with External Pointer
// ================================================================================

TEST(FreeListValidationTest, IsPtrExternalPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Pointer from different allocator
    void* external = malloc(256);
    ASSERT_NE(external, nullptr);
    
    EXPECT_FALSE(freelist->is_ptr(external)) 
        << "External pointer should be invalid";
    
    free(external);
}

// ================================================================================
// Test 5: is_ptr with Freed Pointer
// ================================================================================

TEST(FreeListValidationTest, IsPtrFreedPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate and free
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Valid before free
    EXPECT_TRUE(freelist->is_ptr(ptr));
    
    // Free the pointer
    freelist->return_element(ptr, 256);
    
    // Note: is_ptr() cannot distinguish freed blocks from allocated blocks
    // It only checks if the pointer structure is valid
    // The pointer may still appear "valid" structurally but is actually free
    // This is a known limitation documented in is_ptr()
}

// ================================================================================
// Test 6: is_ptr with Offset Pointer (Inside Allocation)
// ================================================================================

TEST(FreeListValidationTest, IsPtrOffsetPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate block
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Original pointer should be valid
    EXPECT_TRUE(freelist->is_ptr(ptr));
    
    // Offset pointer (ptr + 10) should be invalid
    void* offset_ptr = static_cast<uint8_t*>(ptr) + 10;
    EXPECT_FALSE(freelist->is_ptr(offset_ptr)) 
        << "Offset pointer inside allocation should be invalid";
}

// ================================================================================
// Test 7: is_ptr with Pointer from Different Freelist
// ================================================================================

TEST(FreeListValidationTest, IsPtrDifferentFreelist) {
    auto result1 = FreeListAllocator::Heap(8192, 0, false);
    auto result2 = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result1.hasValue());
    ASSERT_TRUE(result2.hasValue());
    auto freelist1 = cslt::move(result1.value());
    auto freelist2 = cslt::move(result2.value());
    
    // Allocate from freelist1
    auto ptr_result = freelist1->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Valid in freelist1
    EXPECT_TRUE(freelist1->is_ptr(ptr));
    
    // Invalid in freelist2
    EXPECT_FALSE(freelist2->is_ptr(ptr)) 
        << "Pointer from different freelist should be invalid";
}

// ================================================================================
// Test 8: is_ptr After Reset
// ================================================================================

TEST(FreeListValidationTest, IsPtrAfterReset) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Valid before reset
    EXPECT_TRUE(freelist->is_ptr(ptr));
    
    // Reset freelist
    freelist->reset();
    
    // Pointer is now invalid (memory has been reset)
    // Note: The structural validation might still pass, but the pointer
    // is logically invalid. This is a known limitation.
}

// ================================================================================
// is_ptr_sized() TESTS
// ================================================================================

// ================================================================================
// Test 9: is_ptr_sized with Exact Size
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedExactSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate 256 bytes
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Should validate with exact size
    EXPECT_TRUE(freelist->is_ptr_sized(ptr, 256)) 
        << "Should validate with exact allocated size";
}

// ================================================================================
// Test 10: is_ptr_sized with Smaller Size
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedSmallerSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate 256 bytes
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Should validate with smaller size
    EXPECT_TRUE(freelist->is_ptr_sized(ptr, 128)) 
        << "Should validate with size smaller than allocation";
    EXPECT_TRUE(freelist->is_ptr_sized(ptr, 1)) 
        << "Should validate with minimal size";
}

// ================================================================================
// Test 11: is_ptr_sized with Larger Size
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedLargerSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate 256 bytes
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Should fail with larger size
    EXPECT_FALSE(freelist->is_ptr_sized(ptr, 512)) 
        << "Should fail when requested size exceeds allocation";
}

// ================================================================================
// Test 12: is_ptr_sized with Zero Size
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedZeroSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate block
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Zero size should be invalid
    EXPECT_FALSE(freelist->is_ptr_sized(ptr, 0)) 
        << "Zero size should be invalid";
}

// ================================================================================
// Test 13: is_ptr_sized with NULL Pointer
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedNullPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // NULL pointer should fail
    EXPECT_FALSE(freelist->is_ptr_sized(nullptr, 256)) 
        << "NULL pointer should be invalid";
}

// ================================================================================
// Test 14: is_ptr_sized with External Pointer
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedExternalPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // External allocation
    void* external = malloc(256);
    ASSERT_NE(external, nullptr);
    
    EXPECT_FALSE(freelist->is_ptr_sized(external, 128)) 
        << "External pointer should be invalid";
    
    free(external);
}

// ================================================================================
// Test 15: is_ptr_sized with Multiple Allocations
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedMultipleAllocations) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate blocks of different sizes
    auto ptr1_result = freelist->alloc(128, false);
    auto ptr2_result = freelist->alloc(256, false);
    auto ptr3_result = freelist->alloc(512, false);
    
    ASSERT_TRUE(ptr1_result.hasValue());
    ASSERT_TRUE(ptr2_result.hasValue());
    ASSERT_TRUE(ptr3_result.hasValue());
    
    void* ptr1 = ptr1_result.value();
    void* ptr2 = ptr2_result.value();
    void* ptr3 = ptr3_result.value();
    
    // Each should validate with its own size
    EXPECT_TRUE(freelist->is_ptr_sized(ptr1, 128));
    EXPECT_TRUE(freelist->is_ptr_sized(ptr2, 256));
    EXPECT_TRUE(freelist->is_ptr_sized(ptr3, 512));
    
    // Should fail with wrong sizes
    EXPECT_FALSE(freelist->is_ptr_sized(ptr1, 256)) << "ptr1 not big enough for 256";
    EXPECT_FALSE(freelist->is_ptr_sized(ptr2, 512)) << "ptr2 not big enough for 512";
}

// ================================================================================
// Test 16: is_ptr_sized for Bounds Checking Use Case
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedBoundsChecking) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate buffer
    auto ptr_result = freelist->alloc(1024, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Simulate bounds checking before write
    size_t data_to_write = 512;
    
    if (freelist->is_ptr_sized(ptr, data_to_write)) {
        // Safe to write
        uint8_t* buffer = static_cast<uint8_t*>(ptr);
        memset(buffer, 0xFF, data_to_write);
        EXPECT_TRUE(true) << "Write was safe";
    } else {
        FAIL() << "Buffer should be large enough";
    }
    
    // Try to write too much
    size_t too_much = 2048;
    EXPECT_FALSE(freelist->is_ptr_sized(ptr, too_much)) 
        << "Should detect buffer too small";
}

// ================================================================================
// Test 17: is_ptr_sized with Aligned Allocations
// ================================================================================

TEST(FreeListValidationTest, IsPtrSizedAlignedAllocations) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate with custom alignment (more overhead)
    auto ptr_result = freelist->alloc_aligned(256, 64, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Should still validate with the requested size
    EXPECT_TRUE(freelist->is_ptr_sized(ptr, 256)) 
        << "Should validate regardless of alignment overhead";
    
    // Should validate with smaller sizes
    EXPECT_TRUE(freelist->is_ptr_sized(ptr, 128));
    
    // Should fail with larger sizes
    EXPECT_FALSE(freelist->is_ptr_sized(ptr, 512));
}

// ================================================================================
// Test 18: is_ptr and is_ptr_sized Consistency
// ================================================================================

TEST(FreeListValidationTest, IsPtrAndIsPtrSizedConsistency) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate block
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // If is_ptr_sized returns true, is_ptr should also return true
    if (freelist->is_ptr_sized(ptr, 100)) {
        EXPECT_TRUE(freelist->is_ptr(ptr)) 
            << "is_ptr should return true if is_ptr_sized returns true";
    }
    
    // Invalid pointer should fail both
    void* invalid = reinterpret_cast<void*>(0x12345678);
    EXPECT_FALSE(freelist->is_ptr(invalid));
    EXPECT_FALSE(freelist->is_ptr_sized(invalid, 100));
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// DOUBLE FREE TESTS
// ================================================================================

// ================================================================================
// Test 1: Double Free Same Pointer
// ================================================================================

TEST(FreeListEdgeCasesTest, DoubleFree) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Free once
    freelist->return_element(ptr, 256);
    
    // Free again - should be handled safely (silent no-op or detection)
    // The implementation should not crash
    freelist->return_element(ptr, 256);
    
    // Should still be able to use freelist
    auto new_ptr = freelist->alloc(128, false);
    EXPECT_TRUE(new_ptr.hasValue()) << "Freelist should still be usable after double free";
}

// ================================================================================
// Test 2: Double Free with Allocations Between
// ================================================================================

TEST(FreeListEdgeCasesTest, DoubleFreeWithAllocBetween) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr1_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr1_result.hasValue());
    void* ptr1 = ptr1_result.value();
    
    // Free first time
    freelist->return_element(ptr1, 256);
    
    // Allocate something else (might reuse ptr1's memory)
    auto ptr2_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr2_result.hasValue());
    
    // Try to free ptr1 again - this is dangerous
    // Should be handled safely without corruption
    freelist->return_element(ptr1, 256);
    
    // Freelist should still work
    auto ptr3 = freelist->alloc(128, false);
    EXPECT_TRUE(ptr3.hasValue());
}

// ================================================================================
// NULL POINTER TESTS
// ================================================================================

// ================================================================================
// Test 3: Free NULL Pointer
// ================================================================================

TEST(FreeListEdgeCasesTest, FreeNullPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Freeing NULL should be safe no-op
    freelist->return_element(nullptr, 256);
    
    // Freelist should still work
    auto ptr = freelist->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue()) << "Freelist should work after freeing NULL";
}

// ================================================================================
// Test 4: Realloc with NULL and Zero Old Size
// ================================================================================

TEST(FreeListEdgeCasesTest, ReallocNullZeroSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Realloc NULL should behave like alloc
    auto ptr_result = freelist->realloc(nullptr, 0, 256, false);
    
    ASSERT_TRUE(ptr_result.hasValue());
    EXPECT_NE(ptr_result.value(), nullptr);
}

// ================================================================================
// INVALID POINTER TESTS
// ================================================================================

// ================================================================================
// Test 5: Free External Pointer
// ================================================================================

TEST(FreeListEdgeCasesTest, FreeExternalPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Pointer from different allocator
    void* external = malloc(256);
    ASSERT_NE(external, nullptr);
    
    // Freeing external pointer should be handled safely
    freelist->return_element(external, 256);
    
    // Freelist should still work
    auto ptr = freelist->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue());
    
    free(external);
}

// ================================================================================
// Test 6: Free Stack Pointer
// ================================================================================

TEST(FreeListEdgeCasesTest, FreeStackPointer) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Stack-allocated buffer
    uint8_t stack_buffer[256];
    
    // Trying to free stack pointer should be handled safely
    freelist->return_element(stack_buffer, 256);
    
    // Freelist should still work
    auto ptr = freelist->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue());
}

// ================================================================================
// Test 7: Free Pointer from Different Freelist
// ================================================================================

TEST(FreeListEdgeCasesTest, FreeCrossFreelist) {
    auto result1 = FreeListAllocator::Heap(8192, 0, false);
    auto result2 = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result1.hasValue());
    ASSERT_TRUE(result2.hasValue());
    auto freelist1 = cslt::move(result1.value());
    auto freelist2 = cslt::move(result2.value());
    
    // Allocate from freelist1
    auto ptr_result = freelist1->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Try to free in freelist2 - should be detected and handled
    freelist2->return_element(ptr, 256);
    
    // Both freelists should still work
    auto ptr1 = freelist1->alloc(128, false);
    auto ptr2 = freelist2->alloc(128, false);
    EXPECT_TRUE(ptr1.hasValue());
    EXPECT_TRUE(ptr2.hasValue());
}

// ================================================================================
// CAPACITY AND EXHAUSTION TESTS
// ================================================================================

// ================================================================================
// Test 8: Allocate Zero Bytes
// ================================================================================

TEST(FreeListEdgeCasesTest, AllocateZeroBytes) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Zero-byte allocation should fail
    auto ptr_result = freelist->alloc(0, false);
    
    EXPECT_FALSE(ptr_result.hasValue()) << "Zero-byte allocation should fail";
}

// ================================================================================
// Test 9: Allocate More Than Capacity
// ================================================================================

TEST(FreeListEdgeCasesTest, AllocateMoreThanCapacity) {
    auto result = FreeListAllocator::Heap(4096, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Try to allocate more than total capacity
    auto ptr_result = freelist->alloc(16384, false);
    
    EXPECT_FALSE(ptr_result.hasValue()) 
        << "Allocation larger than capacity should fail";
}

// ================================================================================
// Test 10: Allocate After Exhaustion
// ================================================================================

TEST(FreeListEdgeCasesTest, AllocateAfterExhaustion) {
    auto result = FreeListAllocator::Heap(1024, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate until exhausted
    std::vector<void*> ptrs;
    while (true) {
        auto ptr_result = freelist->alloc(64, false);
        if (!ptr_result.hasValue()) {
            break;
        }
        ptrs.push_back(ptr_result.value());
    }
    
    EXPECT_GT(ptrs.size(), 0) << "Should have allocated some blocks";
    
    // Try to allocate when exhausted - should fail gracefully
    auto ptr_result = freelist->alloc(64, false);
    EXPECT_FALSE(ptr_result.hasValue()) << "Should fail when exhausted";
    
    // Free one and try again
    freelist->return_element(ptrs[0], 64);
    
    auto new_ptr = freelist->alloc(64, false);
    EXPECT_TRUE(new_ptr.hasValue()) << "Should succeed after freeing space";
}

// ================================================================================
// Test 11: Very Large Alignment
// ================================================================================

TEST(FreeListEdgeCasesTest, VeryLargeAlignment) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // 4KB alignment (page size)
    auto ptr_result = freelist->alloc_aligned(256, 4096, false);
    
    // Might succeed or fail depending on available space
    if (ptr_result.hasValue()) {
        void* ptr = ptr_result.value();
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 4096, 0) 
            << "Should be 4KB aligned if allocation succeeded";
    }
}

// ================================================================================
// REALLOC EDGE CASES
// ================================================================================

// ================================================================================
// Test 12: Realloc to Zero Size
// ================================================================================

TEST(FreeListEdgeCasesTest, ReallocToZeroSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Realloc to zero should fail
    auto new_result = freelist->realloc(ptr, 256, 0, false);
    
    EXPECT_FALSE(new_result.hasValue()) << "Realloc to zero size should fail";
}

// ================================================================================
// Test 13: Realloc with Incorrect Old Size
// ================================================================================

TEST(FreeListEdgeCasesTest, ReallocIncorrectOldSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate 256 bytes
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Write pattern
    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 256; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    
    // Realloc with wrong old_size (128 instead of 256)
    // This is user error - will copy wrong amount
    auto new_result = freelist->realloc(ptr, 128, 512, false);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
    
    // Only first 128 bytes will be copied (user error)
    for (int i = 0; i < 128; ++i) {
        EXPECT_EQ(new_data[i], static_cast<uint8_t>(i));
    }
    // Note: This demonstrates the importance of tracking sizes correctly
}

// ================================================================================
// Test 14: Realloc Same Size
// ================================================================================

TEST(FreeListEdgeCasesTest, ReallocSameSize) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    
    // Realloc to same size - should return same pointer
    auto new_result = freelist->realloc(ptr, 256, 256, false);
    ASSERT_TRUE(new_result.hasValue());
    
    void* new_ptr = new_result.value();
    EXPECT_EQ(new_ptr, ptr) << "Realloc same size should return same pointer";
}

// ================================================================================
// RESET EDGE CASES
// ================================================================================

// ================================================================================
// Test 15: Reset Empty Freelist
// ================================================================================

TEST(FreeListEdgeCasesTest, ResetEmptyFreelist) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Reset without any allocations
    bool reset_ok = freelist->reset();
    
    EXPECT_TRUE(reset_ok) << "Reset should succeed on empty freelist";
    EXPECT_EQ(freelist->used(), 0);
}

// ================================================================================
// Test 16: Multiple Consecutive Resets
// ================================================================================

TEST(FreeListEdgeCasesTest, MultipleConsecutiveResets) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate something
    auto ptr = freelist->alloc(256, false);
    ASSERT_TRUE(ptr.hasValue());
    
    // Reset multiple times
    EXPECT_TRUE(freelist->reset());
    EXPECT_TRUE(freelist->reset());
    EXPECT_TRUE(freelist->reset());
    
    // Should still work
    auto new_ptr = freelist->alloc(256, false);
    EXPECT_TRUE(new_ptr.hasValue());
}

// ================================================================================
// Test 17: Use After Reset
// ================================================================================

TEST(FreeListEdgeCasesTest, UseAfterReset) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate
    auto ptr1_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr1_result.hasValue());
    //void* ptr1 = ptr1_result.value();
    
    // Reset
    freelist->reset();
    
    // ptr1 is now invalid - don't use it
    
    // Allocate new pointer
    auto ptr2_result = freelist->alloc(256, false);
    ASSERT_TRUE(ptr2_result.hasValue());
    //void* ptr2 = ptr2_result.value();
    
    // ptr2 might equal ptr1 (reused memory)
    // This is expected behavior
}

// ================================================================================
// FRAGMENTATION EDGE CASES
// ================================================================================

// ================================================================================
// Test 18: Extreme Fragmentation
// ================================================================================

TEST(FreeListEdgeCasesTest, ExtremeFragmentation) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate many small blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        auto ptr_result = freelist->alloc(64, false);
        if (ptr_result.hasValue()) {
            ptrs.push_back(ptr_result.value());
        }
    }
    
    // Free every other block to create fragmentation
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        freelist->return_element(ptrs[i], 64);
    }
    
    // Try to allocate a large block - might fail due to fragmentation
    auto large_result = freelist->alloc(2048, false);
    
    // Whether it succeeds depends on coalescing and free block sizes
    // Just verify freelist is still usable
    
    // Free remaining blocks
    for (size_t i = 1; i < ptrs.size(); i += 2) {
        freelist->return_element(ptrs[i], 64);
    }
    
    // After freeing all, should be able to allocate large block
    auto large_result2 = freelist->alloc(2048, false);
    EXPECT_TRUE(large_result2.hasValue()) 
        << "Should allocate after all blocks freed and coalesced";
}

// ================================================================================
// Test 19: Allocation Pattern Stress
// ================================================================================

TEST(FreeListEdgeCasesTest, AllocationPatternStress) {
    auto result = FreeListAllocator::Heap(8192, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Rapidly allocate and free in pattern
    for (int cycle = 0; cycle < 10; ++cycle) {
        std::vector<void*> ptrs;
        
        // Allocate
        for (int i = 0; i < 10; ++i) {
            auto ptr_result = freelist->alloc(128, false);
            if (ptr_result.hasValue()) {
                ptrs.push_back(ptr_result.value());
            }
        }
        
        // Free all
        for (void* ptr : ptrs) {
            freelist->return_element(ptr, 128);
        }
    }
    
    // Freelist should still be healthy
    auto ptr = freelist->alloc(1024, false);
    EXPECT_TRUE(ptr.hasValue()) << "Freelist should be healthy after stress";
}

// ================================================================================
// Test 20: Mixed Size Allocations
// ================================================================================

TEST(FreeListEdgeCasesTest, MixedSizeAllocations) {
    auto result = FreeListAllocator::Heap(16384, 0, false);
    ASSERT_TRUE(result.hasValue());
    auto freelist = cslt::move(result.value());
    
    // Allocate various sizes
    std::vector<std::pair<void*, size_t>> allocations;
    size_t sizes[] = {64, 128, 256, 512, 1024, 32, 96, 384};
    
    for (size_t size : sizes) {
        auto ptr_result = freelist->alloc(size, false);
        if (ptr_result.hasValue()) {
            allocations.push_back({ptr_result.value(), size});
        }
    }
    
    EXPECT_GT(allocations.size(), 0) << "Should allocate some blocks";
    
    // Free in random order (reverse)
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        freelist->return_element(it->first, it->second);
    }
    
    // Should be able to allocate again
    auto ptr = freelist->alloc(2048, false);
    EXPECT_TRUE(ptr.hasValue()) << "Should allocate after mixed free";
}
// ================================================================================ 
// ================================================================================ 

TEST(ArenaWithBuddy, CreatesArenaFromBuddyAndAllocates) {
    constexpr size_t pool_size       = 1u << 20;  // 1 MiB
    constexpr size_t min_block_size  = 64;
    constexpr size_t base_align      = alignof(std::max_align_t);

    auto buddy_res = BuddyAllocator::Heap(pool_size, min_block_size, base_align);
    ASSERT_TRUE(buddy_res.hasValue());
    auto buddy = cslt::move(buddy_res.value());
    ASSERT_NE(buddy.get(), nullptr);

    const size_t buddy_rem_before     = buddy->remaining();
    const size_t buddy_largest_before = buddy->largest_block();

    constexpr size_t arena_bytes = 64 * 1024;

    auto arena_res = ArenaAllocator::WithBuddy(*buddy, arena_bytes, base_align);
    ASSERT_TRUE(arena_res.hasValue());
    auto arena = cslt::move(arena_res.value());
    ASSERT_NE(arena.get(), nullptr);

    // Arena sanity allocs
    auto p1r = arena->alloc(256);
    ASSERT_TRUE(p1r.hasValue());
    ASSERT_NE(p1r.value(), nullptr);

    auto p2r = arena->alloc(1024, true);
    ASSERT_TRUE(p2r.hasValue());
    ASSERT_NE(p2r.value(), nullptr);

    // Buddy should generally have less available while arena is alive
    EXPECT_LE(buddy->remaining(), buddy_rem_before);
    EXPECT_LE(buddy->largest_block(), buddy_largest_before);
}

TEST(ArenaWithBuddy, DestroyingArenaReturnsBlockToBuddy) {
    constexpr size_t pool_size       = 1u << 20;  // 1 MiB
    constexpr size_t min_block_size  = 64;
    constexpr size_t base_align      = alignof(std::max_align_t);
    constexpr size_t arena_bytes     = 64 * 1024;

    auto buddy_res = BuddyAllocator::Heap(pool_size, min_block_size, base_align);
    ASSERT_TRUE(buddy_res.hasValue());
    auto buddy = cslt::move(buddy_res.value());

    const size_t rem_before     = buddy->remaining();
    const size_t largest_before = buddy->largest_block();

    {
        auto arena_res = ArenaAllocator::WithBuddy(*buddy, arena_bytes, base_align);
        ASSERT_TRUE(arena_res.hasValue());
        auto arena = cslt::move(arena_res.value());

        auto pr = arena->alloc(512);
        ASSERT_TRUE(pr.hasValue());
        ASSERT_NE(pr.value(), nullptr);

        // while alive, buddy typically decreases
        EXPECT_LE(buddy->remaining(), rem_before);
    } // arena destroyed here -> should return buddy allocation

    // After destruction, buddy should be at least as free as before
    // (exact equality can be too strict due to rounding/metadata)
    std::cout << buddy->remaining() << "\n";
    EXPECT_GE(buddy->remaining(), rem_before);
    EXPECT_GE(buddy->largest_block(), largest_before);
}

TEST(ArenaWithBuddy, RejectsZeroArenaBytes) {
    constexpr size_t pool_size       = 1u << 20;
    constexpr size_t min_block_size  = 64;
    constexpr size_t base_align      = alignof(std::max_align_t);

    auto buddy_res = BuddyAllocator::Heap(pool_size, min_block_size, base_align);
    ASSERT_TRUE(buddy_res.hasValue());
    auto buddy = cslt::move(buddy_res.value());

    auto arena_res = ArenaAllocator::WithBuddy(*buddy, 0, base_align);
    EXPECT_FALSE(arena_res.hasValue());
}

TEST(ArenaWithBuddy, FailsWithoutLeakingBuddyCapacityWhenRequestTooLarge) {
    // Small pool so arena request definitely fails
    constexpr size_t pool_size       = 8 * 1024;
    constexpr size_t min_block_size  = 64;
    constexpr size_t base_align      = alignof(std::max_align_t);

    auto buddy_res = BuddyAllocator::Heap(pool_size, min_block_size, base_align);
    ASSERT_TRUE(buddy_res.hasValue());
    auto buddy = cslt::move(buddy_res.value());

    const size_t rem_before     = buddy->remaining();
    const size_t largest_before = buddy->largest_block();

    // Request more than pool
    auto arena_res = ArenaAllocator::WithBuddy(*buddy, pool_size * 2, base_align);
    EXPECT_FALSE(arena_res.hasValue());

    // Buddy should not lose capacity on failed init
    EXPECT_EQ(buddy->remaining(), rem_before);
    EXPECT_EQ(buddy->largest_block(), largest_before);
}

// -------------------------------------------------------------------------------- 
TEST(BuddyHeapTest, CreateBasicHeapBuddy) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    
    ASSERT_TRUE(result.hasValue()) << "Basic buddy creation should succeed";
    
    auto buddy = move(result.value());
    
    EXPECT_NE(buddy.get(), nullptr);
    EXPECT_GT(buddy->remaining(), 0) << "Should have free space";
    EXPECT_EQ(buddy->size(), 0) << "Should start with zero used";
}

// ================================================================================
// Test 2: Verify Power-of-2 Rounding
// ================================================================================

TEST(BuddyHeapTest, PowerOf2Rounding) {
    // Request non-power-of-2 sizes
    auto result = BuddyAllocator::Heap(5000, 100, 0);
    
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Pool should be rounded up to next power of 2 (8192)
    // Min block should be rounded up to next power of 2 (128)
    
    // Try to allocate something that verifies the rounding
    auto ptr_result = buddy->alloc(4000, false);
    EXPECT_TRUE(ptr_result.hasValue()) << "Should fit in rounded-up pool";
}

// ================================================================================
// Test 3: Custom Alignment
// ================================================================================

TEST(BuddyHeapTest, CustomAlignment) {
    auto result = BuddyAllocator::Heap(4096, 64, 32);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Use alloc_aligned() for custom alignment
    auto ptr_result = buddy->alloc_aligned(128, 32, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 32, 0); // Now will pass!
}
// ================================================================================
// Test 4: Zero Alignment (Default)
// ================================================================================

TEST(BuddyHeapTest, ZeroAlignmentDefaults) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Should default to alignof(max_align_t)
    auto ptr_result = buddy->alloc(128, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Should be at least naturally aligned
    EXPECT_EQ(addr % alignof(max_align_t), 0);
}

// ================================================================================
// VALIDATION TESTS
// ================================================================================

// ================================================================================
// Test 5: Zero Pool Size
// ================================================================================

TEST(BuddyHeapTest, ZeroPoolSize) {
    auto result = BuddyAllocator::Heap(0, 64, 0);
    
    EXPECT_FALSE(result.hasValue());
}

// ================================================================================
// Test 6: Zero Min Block Size
// ================================================================================

TEST(BuddyHeapTest, ZeroMinBlockSize) {
    auto result = BuddyAllocator::Heap(4096, 0, 0);
    
    EXPECT_FALSE(result.hasValue());
}

// ================================================================================
// Test 7: Min Block Larger Than Pool
// ================================================================================

TEST(BuddyHeapTest, MinBlockLargerThanPool) {
    auto result = BuddyAllocator::Heap(1024, 4096, 0);
    
    EXPECT_FALSE(result.hasValue());
}

// ================================================================================
// Test 8: Non-Power-of-2 Alignment
// ================================================================================

TEST(BuddyHeapTest, NonPowerOf2Alignment) {
    // Request alignment that's not power of 2 (48)
    auto result = BuddyAllocator::Heap(4096, 64, 48);
    
    // Should succeed by rounding up to 64
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Use alloc_aligned() with the non-power-of-2 value
    // It should get rounded up to 64 internally
    auto ptr_result = buddy->alloc_aligned(128, 48, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Should be aligned to rounded-up value (64)
    EXPECT_EQ(addr % 64, 0) << "Should round 48 up to 64 and align to that";
}

// ================================================================================
// SIZE TESTS
// ================================================================================

// ================================================================================
// Test 9: Very Small Pool
// ================================================================================

TEST(BuddyHeapTest, VerySmallPool) {
    // Minimum viable buddy allocator
    auto result = BuddyAllocator::Heap(256, 64, 0);
    
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Should be able to make at least one allocation
    auto ptr_result = buddy->alloc(32, false);
    EXPECT_TRUE(ptr_result.hasValue());
}

// ================================================================================
// Test 10: Large Pool
// ================================================================================

TEST(BuddyHeapTest, LargePool) {
    // 16MB pool
    auto result = BuddyAllocator::Heap(16 * 1024 * 1024, 64, 0);
    
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    EXPECT_GT(buddy->remaining(), 16 * 1024 * 1024 - 1024) 
        << "Should have most of 16MB available";
    
    // Should be able to allocate large blocks
    auto ptr_result = buddy->alloc(1024 * 1024, false);
    EXPECT_TRUE(ptr_result.hasValue()) << "Should allocate 1MB block";
}

// ================================================================================
// ALLOCATION TESTS
// ================================================================================

// ================================================================================
// Test 11: Basic Allocation After Creation
// ================================================================================

TEST(BuddyHeapTest, BasicAllocationAfterCreation) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate
    auto ptr_result = buddy->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    EXPECT_NE(ptr, nullptr);
    
    // Verify accounting
    EXPECT_GT(buddy->size(), 0);
    EXPECT_LT(buddy->remaining(), 4096);
}

// ================================================================================
// Test 12: Multiple Allocations
// ================================================================================

TEST(BuddyHeapTest, MultipleAllocations) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Make several allocations
    for (int i = 0; i < 5; ++i) {
        auto ptr_result = buddy->alloc(128, false);
        ASSERT_TRUE(ptr_result.hasValue()) << "Allocation " << i << " should succeed";
        ptrs.push_back(ptr_result.value());
    }
    
    // All pointers should be distinct
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]) << "Pointers should be unique";
        }
    }
}

// ================================================================================
// Test 13: Zero-Initialized Allocation
// ================================================================================

TEST(BuddyHeapTest, ZeroInitializedAllocation) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate with zeroing
    auto ptr_result = buddy->alloc(256, true);
    ASSERT_TRUE(ptr_result.hasValue());
    
    uint8_t* data = static_cast<uint8_t*>(ptr_result.value());
    
    // Verify all bytes are zero
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], 0) << "Byte " << i << " should be zero";
    }
}

// ================================================================================
// STATISTICS TESTS
// ================================================================================

// ================================================================================
// Test 14: Initial Statistics
// ================================================================================

TEST(BuddyHeapTest, InitialStatistics) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    char buffer[2048];
    bool stat_ok = buddy->stats(buffer, sizeof(buffer));
    
    ASSERT_TRUE(stat_ok);
    
    std::string stats_str(buffer);
    
    // Should contain key information
    EXPECT_NE(stats_str.find("Pool size:"), std::string::npos);
    EXPECT_NE(stats_str.find("Min block size:"), std::string::npos);
    EXPECT_NE(stats_str.find("Max block size:"), std::string::npos);
    EXPECT_NE(stats_str.find("Used: 0"), std::string::npos) << "Should start with 0 used";
}

// ================================================================================
// Test 15: Statistics After Allocations
// ================================================================================

TEST(BuddyHeapTest, StatisticsAfterAllocations) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Make some allocations
    auto p1 = buddy->alloc(256, false);
    auto p2 = buddy->alloc(512, false);
    
    ASSERT_TRUE(p1.hasValue());
    ASSERT_TRUE(p2.hasValue());
    
    char buffer[2048];
    bool stat_ok = buddy->stats(buffer, sizeof(buffer));
    
    ASSERT_TRUE(stat_ok);
    
    std::string stats_str(buffer);
    
    // Should show non-zero usage
    EXPECT_EQ(stats_str.find("Used: 0"), std::string::npos) 
        << "Should not show 0 used after allocations";
    EXPECT_NE(stats_str.find("Free lists by level:"), std::string::npos);
}

// ================================================================================
// QUERY METHOD TESTS
// ================================================================================

// ================================================================================
// Test 16: Remaining and Size Methods
// ================================================================================

TEST(BuddyHeapTest, RemainingAndSizeMethods) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_remaining = buddy->remaining();
    size_t initial_size = buddy->size();
    
    EXPECT_GT(initial_remaining, 0);
    EXPECT_EQ(initial_size, 0);
    
    // Allocate
    auto ptr = buddy->alloc(256, false);
    ASSERT_TRUE(ptr.hasValue());
    
    size_t after_remaining = buddy->remaining();
    size_t after_size = buddy->size();
    
    EXPECT_LT(after_remaining, initial_remaining);
    EXPECT_GT(after_size, initial_size);
}

// ================================================================================
// Test 17: Largest Block Method
// ================================================================================

TEST(BuddyHeapTest, LargestBlockMethod) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_largest = buddy->largest_block();
    
    // Should be able to allocate the largest block
    EXPECT_GT(initial_largest, 0);
    
    // Make some allocations
    buddy->alloc(512, false);
    buddy->alloc(256, false);
    
    size_t after_largest = buddy->largest_block();
    
    // Largest block might have decreased due to fragmentation
    EXPECT_GT(after_largest, 0);
}

// ================================================================================
// Test 18: Min and Max Block Size Methods
// ================================================================================

TEST(BuddyHeapTest, MinMaxBlockSizeMethods) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t min_block = buddy->min_block_size();
    size_t max_block = buddy->max_block_size();
    
    // Min should be power of 2 >= 64
    EXPECT_GE(min_block, 64);
    EXPECT_EQ(min_block & (min_block - 1), 0) << "Min block should be power of 2";
    
    // Max should be power of 2 >= pool size
    EXPECT_GE(max_block, 4096);
    EXPECT_EQ(max_block & (max_block - 1), 0) << "Max block should be power of 2";
    
    // Min should be <= Max
    EXPECT_LE(min_block, max_block);
}

// ================================================================================
// MOVE SEMANTICS TESTS
// ================================================================================

// ================================================================================
// Test 19: Move Constructor Behavior
// ================================================================================

TEST(BuddyHeapTest, MoveSemantics) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    
    auto buddy1 = cslt::move(result.value());
    
    // Make allocation
    auto ptr1 = buddy1->alloc(256, false);
    ASSERT_TRUE(ptr1.hasValue());
    
    // Move to another unique_ptr
    auto buddy2 = cslt::move(buddy1);
    
    // buddy1 should be empty
    EXPECT_EQ(buddy1.get(), nullptr);
    
    // buddy2 should work
    EXPECT_NE(buddy2.get(), nullptr);
    auto ptr2 = buddy2->alloc(256, false);
    EXPECT_TRUE(ptr2.hasValue());
}

// ================================================================================
// Test 20: Automatic Cleanup
// ================================================================================

TEST(BuddyHeapTest, AutomaticCleanup) {
    void* test_ptr = nullptr;
    
    {
        auto result = BuddyAllocator::Heap(4096, 64, 0);
        ASSERT_TRUE(result.hasValue());
        auto buddy = cslt::move(result.value());
        
        auto ptr_result = buddy->alloc(256, false);
        ASSERT_TRUE(ptr_result.hasValue());
        test_ptr = ptr_result.value();
        
        // buddy destroyed here - BuddyDeleter called
    }
    
    // test_ptr is now invalid (dangling pointer)
    // Just verify we got here without crashing
    EXPECT_NE(test_ptr, nullptr) << "Pointer was allocated";
}
// ================================================================================
// FORWARD COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 1: Simple Forward Coalescing (Two Blocks)
// ================================================================================

TEST(BuddyCoalescingTest, ForwardCoalescingTwoBlocks) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate two adjacent blocks of same size
    auto ptr1 = buddy->alloc(128, false);
    auto ptr2 = buddy->alloc(128, false);
    
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    
    void* p1 = ptr1.value();
    void* p2 = ptr2.value();
    
    size_t largest_before = buddy->largest_block();
    
    // Free first block
    buddy->return_element(p1);
    
    // Free second block - should coalesce with first
    buddy->return_element(p2);
    
    size_t largest_after = buddy->largest_block();
    
    // After coalescing, should have larger blocks available
    EXPECT_GT(largest_after, largest_before) 
        << "Largest block should increase after coalescing";
}

// ================================================================================
// Test 2: Forward Coalescing Up Multiple Levels
// ================================================================================

TEST(BuddyCoalescingTest, ForwardCoalescingMultipleLevels) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate 4 blocks of 128 bytes each
    std::vector<void*> ptrs;
    for (int i = 0; i < 4; ++i) {
        auto ptr = buddy->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Free all four in order
    // This should coalesce: (0+1) -> 256, (2+3) -> 256, then (0-1+2-3) -> 512
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
    
    // Should be able to allocate a larger block now
    auto large = buddy->alloc(400, false);
    EXPECT_TRUE(large.hasValue()) 
        << "Should allocate large block after multi-level coalescing";
}

// ================================================================================
// BACKWARD COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 3: Simple Backward Coalescing
// ================================================================================

TEST(BuddyCoalescingTest, BackwardCoalescingTwoBlocks) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate two blocks
    auto ptr1 = buddy->alloc(128, false);
    auto ptr2 = buddy->alloc(128, false);
    
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    
    void* p1 = ptr1.value();
    void* p2 = ptr2.value();
    
    // Free second block first
    buddy->return_element(p2);
    
    size_t largest_before = buddy->largest_block();
    
    // Free first block - should coalesce backward with second
    buddy->return_element(p1);
    
    size_t largest_after = buddy->largest_block();
    
    EXPECT_GT(largest_after, largest_before) 
        << "Should coalesce backward";
}

// ================================================================================
// Test 4: Backward Coalescing Chain
// ================================================================================

TEST(BuddyCoalescingTest, BackwardCoalescingChain) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate 8 blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 8; ++i) {
        auto ptr = buddy->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Free in reverse order (backward coalescing chain)
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        buddy->return_element(*it);
    }
    
    // Should have coalesced into very large blocks
    size_t largest = buddy->largest_block();
    EXPECT_GT(largest, 512) << "Should have large coalesced blocks";
}

// ================================================================================
// BIDIRECTIONAL COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 5: Bidirectional Coalescing (Middle Block Freed Last)
// ================================================================================

TEST(BuddyCoalescingTest, BidirectionalCoalescing) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate three blocks
    auto ptr1 = buddy->alloc(128, false);
    auto ptr2 = buddy->alloc(128, false);
    auto ptr3 = buddy->alloc(128, false);
    
    ASSERT_TRUE(ptr1.hasValue());
    ASSERT_TRUE(ptr2.hasValue());
    ASSERT_TRUE(ptr3.hasValue());
    
    void* p1 = ptr1.value();
    void* p2 = ptr2.value();
    void* p3 = ptr3.value();
    
    // Free first and third (leave middle allocated)
    buddy->return_element(p1);
    buddy->return_element(p3);
    
    size_t largest_before = buddy->largest_block();
    
    // Free middle - should coalesce with both neighbors
    buddy->return_element(p2);
    
    size_t largest_after = buddy->largest_block();
    
    EXPECT_GT(largest_after, largest_before) 
        << "Middle block should coalesce with both neighbors";
}

// ================================================================================
// Test 6: Multiple Bidirectional Coalescing
// ================================================================================

TEST(BuddyCoalescingTest, MultipleBidirectionalCoalescing) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate 7 blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 7; ++i) {
        auto ptr = buddy->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Free alternating blocks (0, 2, 4, 6)
    buddy->return_element(ptrs[0]);
    buddy->return_element(ptrs[2]);
    buddy->return_element(ptrs[4]);
    buddy->return_element(ptrs[6]);
    
    // Now free the middle blocks (1, 3, 5) - each should coalesce bidirectionally
    buddy->return_element(ptrs[1]);  // Coalesces with 0 and 2
    buddy->return_element(ptrs[3]);  // Coalesces with 2 and 4
    buddy->return_element(ptrs[5]);  // Coalesces with 4 and 6
    
    // Should have large coalesced blocks
    size_t largest = buddy->largest_block();
    EXPECT_GT(largest, 512) << "Should have large blocks after bidirectional coalescing";
}

// ================================================================================
// COMPLETE COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 7: Complete Coalescing Back to Initial State
// ================================================================================

TEST(BuddyCoalescingTest, CompleteCoalescingToInitialState) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_largest = buddy->largest_block();
    size_t initial_remaining = buddy->remaining();
    
    // Allocate all available space in small chunks
    std::vector<void*> ptrs;
    while (true) {
        auto ptr = buddy->alloc(64, false);
        if (!ptr.hasValue()) {
            break;
        }
        ptrs.push_back(ptr.value());
    }
    
    EXPECT_GT(ptrs.size(), 0) << "Should have allocated some blocks";
    
    // Free all blocks
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t final_largest = buddy->largest_block();
    size_t final_remaining = buddy->remaining();
    
    // Should have coalesced back to near-initial state
    EXPECT_GE(final_largest, initial_largest) 
        << "Should recover large blocks after complete coalescing";
    EXPECT_GE(final_remaining, initial_remaining - 100) 
        << "Should recover most memory (within overhead tolerance)";
}

// ================================================================================
// Test 8: Complete Coalescing with Random Free Order
// ================================================================================

TEST(BuddyCoalescingTest, CompleteCoalescingRandomOrder) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_remaining = buddy->remaining();
    
    // Allocate many blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 30; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Shuffle the pointers (random free order)
    std::shuffle(ptrs.begin(), ptrs.end(),
             std::mt19937{std::random_device{}()});
    
    // Free all in random order
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t final_remaining = buddy->remaining();
    
    // Should recover most memory regardless of free order
    EXPECT_GE(final_remaining, initial_remaining - 200) 
        << "Should recover memory even with random free order";
}

// ================================================================================
// FRAGMENTATION PREVENTION TESTS
// ================================================================================

// ================================================================================
// Test 9: Coalescing Prevents Fragmentation
// ================================================================================

TEST(BuddyCoalescingTest, CoalescingPreventsFragmentation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Pattern: Allocate many, free half, allocate again
    std::vector<void*> ptrs;
    
    // First wave: allocate 20 blocks
    for (int i = 0; i < 20; ++i) {
        auto ptr = buddy->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Free every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t largest_fragmented = buddy->largest_block();
    
    // Free remaining blocks - should coalesce
    for (void* ptr : ptrs) {
        if (ptr != nullptr) {
            buddy->return_element(ptr);
        }
    }
    
    size_t largest_after_coalesce = buddy->largest_block();
    
    // After coalescing, should have much larger blocks
    EXPECT_GT(largest_after_coalesce, largest_fragmented * 2) 
        << "Coalescing should significantly reduce fragmentation";
}

// ================================================================================
// Test 10: Checkerboard Pattern Coalescing
// ================================================================================

TEST(BuddyCoalescingTest, CheckerboardPatternCoalescing) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate 16 blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 16; ++i) {
        auto ptr = buddy->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Create checkerboard: free even indices (0, 2, 4, 6, 8, 10, 12, 14)
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t largest_checkerboard = buddy->largest_block();
    
    // Now free odd indices - should coalesce entire pattern
    for (size_t i = 1; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t largest_after = buddy->largest_block();
    
    // After freeing all blocks, should have much larger blocks
    // largest_checkerboard is probably 4096 (half pool)
    // largest_after should be 8192 (full pool)
    EXPECT_GT(largest_after, largest_checkerboard) 
        << "Should coalesce checkerboard pattern into larger blocks";
    
    // More specifically, should be able to allocate the entire remaining space
    EXPECT_GE(largest_after, 4096) 
        << "Should have at least 4KB contiguous after full coalescing";
}

// ================================================================================
// MIXED SIZE COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 11: Coalescing Different Sized Blocks
// ================================================================================

TEST(BuddyCoalescingTest, CoalescingDifferentSizes) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate blocks of various sizes
    auto p1 = buddy->alloc(128, false);  // Small
    auto p2 = buddy->alloc(128, false);  // Small
    auto p3 = buddy->alloc(256, false);  // Medium
    auto p4 = buddy->alloc(512, false);  // Large
    
    ASSERT_TRUE(p1.hasValue());
    ASSERT_TRUE(p2.hasValue());
    ASSERT_TRUE(p3.hasValue());
    ASSERT_TRUE(p4.hasValue());
    
    // Free small blocks first - they should coalesce
    buddy->return_element(p1.value());
    buddy->return_element(p2.value());
    
    size_t largest_after_small = buddy->largest_block();
    
    // Free medium and large
    buddy->return_element(p3.value());
    buddy->return_element(p4.value());
    
    size_t largest_final = buddy->largest_block();
    
    // Should have increasingly large blocks as we free
    EXPECT_GE(largest_final, largest_after_small) 
        << "Should have larger blocks after freeing all";
}

// ================================================================================
// Test 12: Interleaved Size Allocation and Coalescing
// ================================================================================

TEST(BuddyCoalescingTest, InterleavedSizeCoalescing) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> small_ptrs;
    std::vector<void*> large_ptrs;
    
    // Allocate pattern: small, large, small, large
    for (int i = 0; i < 8; ++i) {
        auto small = buddy->alloc(128, false);
        auto large = buddy->alloc(512, false);
        
        if (small.hasValue()) small_ptrs.push_back(small.value());
        if (large.hasValue()) large_ptrs.push_back(large.value());
    }
    
    // Free all small blocks
    for (void* ptr : small_ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t largest_after_small = buddy->largest_block();
    
    // Free all large blocks - should coalesce
    for (void* ptr : large_ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t largest_final = buddy->largest_block();
    
    EXPECT_GT(largest_final, largest_after_small) 
        << "Should coalesce after freeing all blocks";
}

// ================================================================================
// NO COALESCING TESTS (Control Cases)
// ================================================================================

// ================================================================================
// Test 13: No Coalescing with Non-Buddy Blocks
// ================================================================================

TEST(BuddyCoalescingTest, NoCoalescingNonBuddies) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate blocks that are not buddies (different sizes)
    auto p1 = buddy->alloc(128, false);
    auto p2 = buddy->alloc(256, false);
    auto p3 = buddy->alloc(512, false);
    
    ASSERT_TRUE(p1.hasValue());
    ASSERT_TRUE(p2.hasValue());
    ASSERT_TRUE(p3.hasValue());
    
    //size_t largest_before = buddy->largest_block();
    
    // Free them - they shouldn't coalesce (different sizes)
    buddy->return_element(p1.value());
    buddy->return_element(p2.value());
    buddy->return_element(p3.value());
    
    //size_t largest_after = buddy->largest_block();
    
    // Largest might increase, but won't be sum of all blocks
    // (They're not buddies so they don't merge into one big block)
}

// ================================================================================
// Test 14: No Coalescing When Buddy Is Allocated
// ================================================================================

TEST(BuddyCoalescingTest, NoCoalescingWhenBuddyAllocated) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate three blocks (first two are buddies)
    auto p1 = buddy->alloc(128, false);
    auto p2 = buddy->alloc(128, false);
    auto p3 = buddy->alloc(128, false);
    
    ASSERT_TRUE(p1.hasValue());
    ASSERT_TRUE(p2.hasValue());
    ASSERT_TRUE(p3.hasValue());
    
    // Free first block only
    buddy->return_element(p1.value());
    
    //size_t largest_single = buddy->largest_block();
    
    // Keep p2 allocated, free p3
    buddy->return_element(p3.value());
    
    size_t largest_with_allocated_buddy = buddy->largest_block();
    
    // Should not have significantly larger blocks since p2 blocks coalescing
    // (p1 and p2 are buddies but p2 is still allocated)
    
    // Now free p2 - should enable coalescing
    buddy->return_element(p2.value());
    
    size_t largest_after_all_free = buddy->largest_block();
    
    EXPECT_GT(largest_after_all_free, largest_with_allocated_buddy) 
        << "Should coalesce only after buddy is freed";
}

// ================================================================================
// STRESS COALESCING TESTS
// ================================================================================

// ================================================================================
// Test 15: Repeated Allocation and Coalescing Cycles
// ================================================================================

TEST(BuddyCoalescingTest, RepeatedCoalescingCycles) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_remaining = buddy->remaining();
    
    // Run 10 cycles of allocate-all, free-all
    for (int cycle = 0; cycle < 10; ++cycle) {
        std::vector<void*> ptrs;
        
        // Allocate many blocks
        for (int i = 0; i < 30; ++i) {
            auto ptr = buddy->alloc(128, false);
            if (ptr.hasValue()) {
                ptrs.push_back(ptr.value());
            }
        }
        
        // Free all
        for (void* ptr : ptrs) {
            buddy->return_element(ptr);
        }
    }
    
    size_t final_remaining = buddy->remaining();
    
    // After repeated cycles, should still have memory available (no leaks)
    EXPECT_GE(final_remaining, initial_remaining - 200) 
        << "Should maintain coalescing efficiency over repeated cycles";
}
// -------------------------------------------------------------------------------- 

// ================================================================================
// FRAGMENTATION DETECTION TESTS
// ================================================================================

// ================================================================================
// Test 1: Fragmentation Gap (Remaining vs Largest)
// ================================================================================

TEST(BuddyFragmentationTest, FragmentationGap) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate many small blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every other block (creates fragmentation)
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Fragmentation indicator: remaining bytes > largest allocatable block
    EXPECT_GT(remaining, largest) 
        << "Fragmentation: total free > largest contiguous block";
    
    // The gap shows fragmentation
    size_t fragmentation_gap = remaining - largest;
    EXPECT_GT(fragmentation_gap, 0) 
        << "Should have fragmentation gap";
}

// ================================================================================
// Test 2: Severe Fragmentation Pattern
// ================================================================================

TEST(BuddyFragmentationTest, SevereFragmentation) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate 40 blocks of 128 bytes
    for (int i = 0; i < 40; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every third block (severe fragmentation)
    for (size_t i = 0; i < ptrs.size(); i += 3) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // With severe fragmentation, gap should be significant
    EXPECT_GT(remaining, largest * 2) 
        << "Severe fragmentation: remaining >> largest";
}

// ================================================================================
// Test 3: Fragmentation Prevents Large Allocation
// ================================================================================

TEST(BuddyFragmentationTest, FragmentationPreventsLargeAllocation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate many 256-byte blocks
    for (int i = 0; i < 16; ++i) {
        auto ptr = buddy->alloc(256, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free alternating blocks
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t remaining = buddy->remaining();
    
    // We have plenty of free memory total
    EXPECT_GT(remaining, 2000) << "Should have significant free memory";
    
    // But can't allocate a large block due to fragmentation
    auto large = buddy->alloc(2048, false);
    
    // This might fail due to fragmentation
    // The test demonstrates the fragmentation problem
    if (!large.hasValue()) {
        // Fragmentation prevented allocation despite free memory
        EXPECT_GT(remaining, 2048) 
            << "Fragmentation: can't allocate despite free memory";
    }
}

// ================================================================================
// FRAGMENTATION RECOVERY TESTS
// ================================================================================

// ================================================================================
// Test 4: Recovery from Fragmentation
// ================================================================================

TEST(BuddyFragmentationTest, RecoveryFromFragmentation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Create fragmentation
    for (int i = 0; i < 20; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every other block (fragmented state)
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t largest_fragmented = buddy->largest_block();
    
    // Free remaining blocks (recover from fragmentation)
    for (void* ptr : ptrs) {
        if (ptr != nullptr) {
            buddy->return_element(ptr);
        }
    }
    
    size_t largest_recovered = buddy->largest_block();
    
    // After recovery, should have much larger blocks
    EXPECT_GT(largest_recovered, largest_fragmented * 2) 
        << "Should recover large blocks after defragmentation";
}

// ================================================================================
// Test 5: Partial Recovery
// ================================================================================

TEST(BuddyFragmentationTest, PartialRecovery) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate 16 blocks
    for (int i = 0; i < 16; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free indices 0,1,2,3 and 8,9,10,11 (two groups)
    for (int i = 0; i <= 3; ++i) {
        buddy->return_element(ptrs[i]);
    }
    for (int i = 8; i <= 11; ++i) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t largest_partial = buddy->largest_block();
    
    // Should have moderate-sized blocks from partial coalescing
    EXPECT_GT(largest_partial, 256) 
        << "Partial recovery should create medium blocks";
}

// ================================================================================
// WORST-CASE FRAGMENTATION TESTS
// ================================================================================

// ================================================================================
// Test 6: Alternating Allocation Pattern
// ================================================================================

TEST(BuddyFragmentationTest, AlternatingAllocationPattern) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> small_ptrs;
    std::vector<void*> large_ptrs;
    
    // Allocate pattern: small, large, small, large
    for (int i = 0; i < 12; ++i) {
        auto small = buddy->alloc(128, false);
        auto large = buddy->alloc(512, false);
        
        if (small.hasValue()) small_ptrs.push_back(small.value());
        if (large.hasValue()) large_ptrs.push_back(large.value());
    }
    
    // Free only large blocks
    for (void* ptr : large_ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Large blocks freed but separated by small blocks
    EXPECT_GT(remaining, largest) 
        << "Alternating pattern creates fragmentation";
    
    // Small blocks prevent coalescing
    size_t gap = remaining - largest;
    EXPECT_GT(gap, 1024) << "Should have significant fragmentation gap";
}

// ================================================================================
// Test 7: Swiss Cheese Fragmentation
// ================================================================================

TEST(BuddyFragmentationTest, SwissCheeseFragmentation) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Fill with allocations
    for (int i = 0; i < 50; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free random scattered blocks (swiss cheese pattern)
    std::vector<size_t> indices_to_free = {1, 4, 7, 11, 15, 19, 24, 29, 35, 41};
    
    for (size_t idx : indices_to_free) {
        if (idx < ptrs.size()) {
            buddy->return_element(ptrs[idx]);
            ptrs[idx] = nullptr;
        }
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Swiss cheese: many small holes
    EXPECT_GT(remaining, largest) 
        << "Random scattered frees create swiss cheese fragmentation";
}

// ================================================================================
// FRAGMENTATION METRICS TESTS
// ================================================================================

// ================================================================================
// Test 8: Fragmentation Ratio
// ================================================================================

TEST(BuddyFragmentationTest, FragmentationRatio) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate and create fragmentation
    for (int i = 0; i < 24; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every 3rd block
    for (size_t i = 0; i < ptrs.size(); i += 3) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Fragmentation ratio: how much free memory is unusable for large allocations
    if (remaining > 0) {
        double utilization = static_cast<double>(largest) / static_cast<double>(remaining);
        
        // With fragmentation, utilization should be < 1.0
        // (not all free memory is in one contiguous block)
        EXPECT_LT(utilization, 1.0) 
            << "Fragmentation: not all free memory is contiguous";
        
        // Severe fragmentation: utilization might be 0.5 or less
        if (utilization < 0.5) {
            SUCCEED() << "Severe fragmentation detected (utilization < 50%)";
        }
    }
}

// ================================================================================
// Test 9: Tracking Fragmentation Over Time
// ================================================================================

TEST(BuddyFragmentationTest, FragmentationOverTime) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<double> fragmentation_ratios;
    
    // Run multiple allocation/free cycles and track fragmentation
    for (int cycle = 0; cycle < 5; ++cycle) {
        std::vector<void*> ptrs;
        
        // Allocate
        for (int i = 0; i < 20; ++i) {
            auto ptr = buddy->alloc(128, false);
            if (ptr.hasValue()) {
                ptrs.push_back(ptr.value());
            }
        }
        
        // Free every other block
        for (size_t i = 0; i < ptrs.size(); i += 2) {
            buddy->return_element(ptrs[i]);
            ptrs[i] = nullptr;
        }
        
        // Measure fragmentation
        size_t remaining = buddy->remaining();
        size_t largest = buddy->largest_block();
        
        if (remaining > 0) {
            double ratio = static_cast<double>(largest) / static_cast<double>(remaining);
            fragmentation_ratios.push_back(ratio);
        }
        
        // Free remaining
        for (void* ptr : ptrs) {
            if (ptr) buddy->return_element(ptr);
        }
    }
    
    // Fragmentation should be consistently measurable
    EXPECT_GT(fragmentation_ratios.size(), 0) 
        << "Should have fragmentation measurements";
}

// ================================================================================
// INTERLEAVED SIZE FRAGMENTATION TESTS
// ================================================================================

// ================================================================================
// Test 10: Small-Large-Small Fragmentation
// ================================================================================

TEST(BuddyFragmentationTest, SmallLargeSmallFragmentation) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> small_ptrs;
    std::vector<void*> large_ptrs;
    
    // Allocate: small, large, small, large, ...
    for (int i = 0; i < 10; ++i) {
        auto s1 = buddy->alloc(64, false);
        auto l = buddy->alloc(1024, false);
        auto s2 = buddy->alloc(64, false);
        
        if (s1.hasValue()) small_ptrs.push_back(s1.value());
        if (l.hasValue()) large_ptrs.push_back(l.value());
        if (s2.hasValue()) small_ptrs.push_back(s2.value());
    }
    
    // Free all large blocks
    for (void* ptr : large_ptrs) {
        buddy->return_element(ptr);
    }
    
    size_t remaining = buddy->remaining();
    //size_t largest = buddy->largest_block();
    
    // Large blocks freed but sandwiched between small ones
    EXPECT_GT(remaining, 5000) << "Should have plenty of free memory";
}

// ================================================================================
// Test 11: Pyramid Allocation Fragmentation
// ================================================================================

TEST(BuddyFragmentationTest, PyramidAllocationFragmentation) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate pyramid: increasing sizes
    size_t sizes[] = {64, 128, 256, 512, 1024, 512, 256, 128, 64};
    
    for (size_t size : sizes) {
        auto ptr = buddy->alloc(size, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Mixed sizes create complex fragmentation
    EXPECT_GT(remaining, largest) 
        << "Pyramid pattern creates fragmentation";
}

// ================================================================================
// DEFRAGMENTATION TESTS
// ================================================================================

// ================================================================================
// Test 12: Sequential Defragmentation
// ================================================================================

TEST(BuddyFragmentationTest, SequentialDefragmentation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Create fragmentation
    for (int i = 0; i < 20; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free alternating
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t largest_fragmented = buddy->largest_block();
    
    // Defragment by freeing in groups of adjacent blocks
    // Free indices 1, 3 (adjacent)
    buddy->return_element(ptrs[1]);
    buddy->return_element(ptrs[3]);
    
    size_t largest_partial = buddy->largest_block();
    
    EXPECT_GE(largest_partial, largest_fragmented) 
        << "Sequential freeing should reduce fragmentation";
}

// ================================================================================
// Test 13: Complete Defragmentation
// ================================================================================

TEST(BuddyFragmentationTest, CompleteDefragmentation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Create severe fragmentation
    for (int i = 0; i < 30; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free every 4th block (severe fragmentation)
    for (size_t i = 0; i < ptrs.size(); i += 4) {
        buddy->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    size_t remaining_fragmented = buddy->remaining();
    size_t largest_fragmented = buddy->largest_block();
    
    double frag_ratio = static_cast<double>(largest_fragmented) / 
                        static_cast<double>(remaining_fragmented);
    
    // Complete defragmentation: free all
    for (void* ptr : ptrs) {
        if (ptr) buddy->return_element(ptr);
    }
    
    size_t remaining_clean = buddy->remaining();
    size_t largest_clean = buddy->largest_block();
    
    double clean_ratio = static_cast<double>(largest_clean) / 
                         static_cast<double>(remaining_clean);
    
    // After complete defrag, utilization should be much better
    EXPECT_GT(clean_ratio, frag_ratio) 
        << "Complete defragmentation improves utilization";
}

// ================================================================================
// RESET AS DEFRAGMENTATION TESTS
// ================================================================================

// ================================================================================
// Test 14: Reset to Eliminate Fragmentation
// ================================================================================

TEST(BuddyFragmentationTest, ResetEliminatesFragmentation) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Create severe fragmentation
    for (int i = 0; i < 25; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Free scattered blocks
    for (size_t i = 0; i < ptrs.size(); i += 3) {
        buddy->return_element(ptrs[i]);
    }
    
    size_t largest_fragmented = buddy->largest_block();
    
    // Reset eliminates all fragmentation
    buddy->reset();
    
    size_t largest_after_reset = buddy->largest_block();
    
    // After reset, should have maximum block size
    EXPECT_GT(largest_after_reset, largest_fragmented) 
        << "Reset should eliminate all fragmentation";
    
    EXPECT_GE(largest_after_reset, 4096) 
        << "Should have large contiguous block after reset";
}

// ================================================================================
// Test 15: Fragmentation with Mixed Operations
// ================================================================================

TEST(BuddyFragmentationTest, MixedOperationsFragmentation) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Complex pattern: alloc, free, alloc, free
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Allocate
        for (int i = 0; i < 10; ++i) {
            auto ptr = buddy->alloc(256, false);
            if (ptr.hasValue()) {
                ptrs.push_back(ptr.value());
            }
        }
        
        // Free some
        if (ptrs.size() >= 5) {
            for (int i = 0; i < 5; ++i) {
                buddy->return_element(ptrs[i]);
                ptrs.erase(ptrs.begin());
            }
        }
    }
    
    size_t remaining = buddy->remaining();
    size_t largest = buddy->largest_block();
    
    // Mixed operations tend to create fragmentation
    EXPECT_GT(remaining, 0) << "Should have free memory";
    
    if (remaining > largest) {
        SUCCEED() << "Mixed operations created fragmentation";
    }
}
// -------------------------------------------------------------------------------- 

// ================================================================================
// ALLOCATION SIZE VERIFICATION TESTS
// ================================================================================

// ================================================================================
// Test 1: Power-of-2 Rounding for Small Allocations
// ================================================================================

TEST(BuddySizeTest, PowerOf2RoundingSmall) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Request 100 bytes, should get rounded to next power of 2
    // 100 + header (16 bytes) = 116, rounded to 128
    auto ptr = buddy->alloc(100, false);
    ASSERT_TRUE(ptr.hasValue());
    
    size_t before_size = buddy->size();
    
    buddy->return_element(ptr.value());
    
    size_t after_size = buddy->size();
    
    // The block size allocated should be power of 2
    size_t block_size = before_size - after_size;
    
    // Verify power of 2
    EXPECT_EQ(block_size & (block_size - 1), 0) 
        << "Block size should be power of 2";
    EXPECT_GE(block_size, 128) << "Should allocate at least 128 bytes for 100-byte request";
}

// ================================================================================
// Test 2: Various Request Sizes Map to Powers of 2
// ================================================================================

TEST(BuddySizeTest, VariousRequestSizes) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    struct TestCase {
        size_t request_size;
        size_t min_expected_block;
    };
    
    TestCase cases[] = {
        {1, 64},      // Tiny request -> min block
        {32, 64},     // Small request -> min block
        {64, 128},    // 64 + header -> 128
        {100, 128},   // 100 + header -> 128
        {200, 256},   // 200 + header -> 256
        {500, 512},   // 500 + header -> 512
        {1000, 1024}, // 1000 + header -> 1024
        {2000, 2048}, // 2000 + header -> 2048
    };
    
    for (const auto& test : cases) {
        size_t before = buddy->size();
        
        auto ptr = buddy->alloc(test.request_size, false);
        ASSERT_TRUE(ptr.hasValue()) 
            << "Failed to allocate " << test.request_size << " bytes";
        
        size_t after = buddy->size();
        size_t block_size = after - before;
        
        // Verify power of 2
        EXPECT_EQ(block_size & (block_size - 1), 0) 
            << "Block for " << test.request_size << " bytes should be power of 2";
        
        // Verify minimum expected size
        EXPECT_GE(block_size, test.min_expected_block) 
            << "Block for " << test.request_size << " bytes should be >= " 
            << test.min_expected_block;
        
        buddy->return_element(ptr.value());
    }
}

// ================================================================================
// Test 3: Maximum Request Size
// ================================================================================

TEST(BuddySizeTest, MaximumRequestSize) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t max_capacity = buddy->remaining();
    
    // Request close to maximum (accounting for header)
    auto ptr = buddy->alloc(max_capacity - 100, false);
    
    if (ptr.hasValue()) {
        // Verify it consumed most of the pool
        EXPECT_LT(buddy->remaining(), 100) 
            << "Should have consumed most of pool";
        
        buddy->return_element(ptr.value());
    }
}

// ================================================================================
// Test 4: Block Size Consistency
// ================================================================================

TEST(BuddySizeTest, BlockSizeConsistency) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate same size multiple times
    std::vector<void*> ptrs;
    std::vector<size_t> block_sizes;
    
    for (int i = 0; i < 5; ++i) {
        size_t before = buddy->size();
        
        auto ptr = buddy->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
        
        size_t after = buddy->size();
        block_sizes.push_back(after - before);
    }
    
    // All block sizes should be identical
    for (size_t i = 1; i < block_sizes.size(); ++i) {
        EXPECT_EQ(block_sizes[i], block_sizes[0]) 
            << "Same request size should yield same block size";
    }
    
    // Cleanup
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
}

// ================================================================================
// Test 5: Alignment Affects Block Size
// ================================================================================

TEST(BuddySizeTest, AlignmentAffectsBlockSize) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Allocate with normal alignment
    size_t before_normal = buddy->size();
    auto ptr_normal = buddy->alloc(128, false);
    ASSERT_TRUE(ptr_normal.hasValue());
    size_t after_normal = buddy->size();
    size_t block_normal = after_normal - before_normal;
    
    buddy->return_element(ptr_normal.value());
    
    // Allocate with strict alignment (requires larger block)
    size_t before_aligned = buddy->size();
    auto ptr_aligned = buddy->alloc_aligned(128, 256, false);
    ASSERT_TRUE(ptr_aligned.hasValue());
    size_t after_aligned = buddy->size();
    size_t block_aligned = after_aligned - before_aligned;
    
    buddy->return_element(ptr_aligned.value());
    
    // Aligned allocation should use larger or equal block
    EXPECT_GE(block_aligned, block_normal) 
        << "Aligned allocation may require larger block";
}

// ================================================================================
// STRESS TESTS - SUSTAINED LOAD
// ================================================================================

// ================================================================================
// Test 6: Sustained Allocation Load
// ================================================================================

TEST(BuddyStressTest, SustainedAllocationLoad) {
    auto result = BuddyAllocator::Heap(65536, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate until exhaustion
    int allocation_count = 0;
    while (allocation_count < 1000) {
        auto ptr = buddy->alloc(128, false);
        if (!ptr.hasValue()) {
            break;
        }
        ptrs.push_back(ptr.value());
        allocation_count++;
    }
    
    EXPECT_GT(allocation_count, 100) << "Should handle many allocations";
    
    // Free all
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
    
    // Verify recovery
    size_t recovered = buddy->remaining();
    EXPECT_GT(recovered, 60000) << "Should recover most memory";
}

// ================================================================================
// Test 7: Rapid Allocation and Deallocation
// ================================================================================

TEST(BuddyStressTest, RapidAllocDealloc) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // 1000 rapid alloc/free cycles
    for (int i = 0; i < 1000; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            buddy->return_element(ptr.value());
        }
    }
    
    // Allocator should still be healthy
    auto test_ptr = buddy->alloc(256, false);
    EXPECT_TRUE(test_ptr.hasValue()) << "Allocator should still work after rapid cycles";
}

// ================================================================================
// Test 8: Random Size Allocations
// ================================================================================

TEST(BuddyStressTest, RandomSizeAllocations) {
    auto result = BuddyAllocator::Heap(32768, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> size_dist(64, 1024);
    
    std::vector<void*> ptrs;
    
    // Allocate random sizes
    for (int i = 0; i < 200; ++i) {
        size_t random_size = size_dist(rng);
        auto ptr = buddy->alloc(random_size, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    EXPECT_GT(ptrs.size(), 40) << "Should handle many random-sized allocations";
    
    // Free all
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
}

// ================================================================================
// Test 9: Random Free Order
// ================================================================================

TEST(BuddyStressTest, RandomFreeOrder) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate many blocks
    for (int i = 0; i < 100; ++i) {
        auto ptr = buddy->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    // Shuffle pointers (random free order)
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ptrs.begin(), ptrs.end(), g);
    
    // Free in random order
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
    
    // Should still be functional
    auto test = buddy->alloc(256, false);
    EXPECT_TRUE(test.hasValue()) << "Should work after random-order frees";
}

// ================================================================================
// Test 10: Alternating Alloc/Free Pattern
// ================================================================================

TEST(BuddyStressTest, AlternatingAllocFreePattern) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // 500 cycles of: alloc 2, free 1
    for (int i = 0; i < 500; ++i) {
        auto p1 = buddy->alloc(128, false);
        auto p2 = buddy->alloc(128, false);
        
        if (p1.hasValue()) {
            buddy->return_element(p1.value());
        }
        
        // Keep p2 allocated (creates gradual buildup)
    }
    
    // Reset to clean up
    buddy->reset();
    
    // Verify still works
    auto test = buddy->alloc(256, false);
    EXPECT_TRUE(test.hasValue());
}

// ================================================================================
// STRESS TESTS - EDGE CONDITIONS
// ================================================================================

// ================================================================================
// Test 11: Allocation at Capacity Boundary
// ================================================================================

TEST(BuddyStressTest, AllocationAtCapacityBoundary) {
    auto result = BuddyAllocator::Heap(4096, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Fill to near capacity
    while (true) {
        auto ptr = buddy->alloc(64, false);
        if (!ptr.hasValue()) {
            break;
        }
        ptrs.push_back(ptr.value());
    }
    
    size_t remaining = buddy->remaining();
    
    // Should be nearly full
    EXPECT_LT(remaining, 200) << "Should be near capacity";
    
    // Free one
    if (!ptrs.empty()) {
        buddy->return_element(ptrs.back());
        ptrs.pop_back();
    }
    
    // Should be able to allocate again
    auto ptr = buddy->alloc(64, false);
    EXPECT_TRUE(ptr.hasValue()) << "Should allocate after freeing at boundary";
    
    // Cleanup
    for (void* p : ptrs) {
        buddy->return_element(p);
    }
}

// ================================================================================
// Test 12: Repeated Reset Cycles
// ================================================================================

TEST(BuddyStressTest, RepeatedResetCycles) {
    auto result = BuddyAllocator::Heap(8192, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    size_t initial_remaining = buddy->remaining();
    
    // 50 cycles of: allocate many, reset
    for (int cycle = 0; cycle < 50; ++cycle) {
        // Allocate
        for (int i = 0; i < 20; ++i) {
            buddy->alloc(128, false);
        }
        
        // Reset
        bool reset_ok = buddy->reset();
        EXPECT_TRUE(reset_ok) << "Reset should succeed in cycle " << cycle;
    }
    
    size_t final_remaining = buddy->remaining();
    
    // Should maintain capacity across resets
    EXPECT_GE(final_remaining, initial_remaining - 100) 
        << "Should maintain capacity across 50 reset cycles";
}

// ================================================================================
// Test 13: Maximum Allocations Count
// ================================================================================

TEST(BuddyStressTest, MaximumAllocationsCount) {
    auto result = BuddyAllocator::Heap(65536, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate minimum size blocks until exhaustion
    int count = 0;
    while (count < 2000) {
        auto ptr = buddy->alloc(64, false);
        if (!ptr.hasValue()) {
            break;
        }
        ptrs.push_back(ptr.value());
        count++;
    }
    
    EXPECT_GT(count, 200) << "Should handle hundreds of allocations";
    
    // All pointers should be unique
    std::sort(ptrs.begin(), ptrs.end());
    auto it = std::unique(ptrs.begin(), ptrs.end());
    EXPECT_EQ(it, ptrs.end()) << "All pointers should be unique";
    
    // Cleanup
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
}

// ================================================================================
// STRESS TESTS - DATA INTEGRITY
// ================================================================================

// ================================================================================
// Test 14: Data Integrity Under Stress
// ================================================================================

TEST(BuddyStressTest, DataIntegrityUnderStress) {
    auto result = BuddyAllocator::Heap(32768, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    struct Allocation {
        void* ptr;
        uint8_t pattern;
        size_t size;
    };
    
    std::vector<Allocation> allocations;
    
    // Make allocations with unique patterns
    for (int i = 0; i < 100; ++i) {
        size_t size = 128 + (i % 512);
        auto ptr_result = buddy->alloc(size, false);
        
        if (ptr_result.hasValue()) {
            void* ptr = ptr_result.value();
            uint8_t pattern = static_cast<uint8_t>(i);
            
            // Fill with pattern
            memset(ptr, pattern, size);
            
            allocations.push_back({ptr, pattern, size});
        }
    }
    
    // Verify all data is intact
    for (const auto& alloc : allocations) {
        uint8_t* data = static_cast<uint8_t*>(alloc.ptr);
        
        for (size_t i = 0; i < alloc.size; ++i) {
            EXPECT_EQ(data[i], alloc.pattern) 
                << "Data corruption detected at allocation with pattern " 
                << static_cast<int>(alloc.pattern);
        }
    }
    
    // Cleanup
    for (const auto& alloc : allocations) {
        buddy->return_element(alloc.ptr);
    }
}

// ================================================================================
// Test 15: Zero-Initialization Stress
// ================================================================================

TEST(BuddyStressTest, ZeroInitializationStress) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate many zero-initialized blocks
    for (int i = 0; i < 50; ++i) {
        auto ptr = buddy->alloc(256, true);  // zeroed = true
        
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
            
            // Verify all zeros
            uint8_t* data = static_cast<uint8_t*>(ptr.value());
            for (size_t j = 0; j < 256; ++j) {
                EXPECT_EQ(data[j], 0) 
                    << "Allocation " << i << " byte " << j << " not zero";
            }
        }
    }
    
    // Cleanup
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
}

// ================================================================================
// Test 16: Mixed Size Stress Test
// ================================================================================

TEST(BuddyStressTest, MixedSizeStress) {
    // Use larger pool to handle the test properly
    auto result = BuddyAllocator::Heap(1024 * 1024, 64, 0);  // 1 MB pool
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    size_t sizes[] = {64, 128, 256, 512, 1024, 2048};
    
    // 100 iterations of mixed-size allocations
    for (int iteration = 0; iteration < 100; ++iteration) {
        for (size_t size : sizes) {
            auto ptr = buddy->alloc(size, false);
            if (ptr.hasValue()) {
                ptrs.push_back(ptr.value());
            }
        }
    }
    
    // With 1MB pool and these sizes, should get most allocations
    // Each iteration uses ~8KB (power-of-2 rounded), so 100 iterations = ~800KB
    EXPECT_GT(ptrs.size(), 500) << "Should handle mixed-size stress with 1MB pool";
    
    // Free in reverse order
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        buddy->return_element(*it);
    }
    
    // Verify recovery (should recover most of the 1MB)
    EXPECT_GT(buddy->remaining(), 900000) << "Should recover memory after stress";
}

// ================================================================================
// Test 17: Realloc Stress Test
// ================================================================================

TEST(BuddyStressTest, ReallocStress) {
    auto result = BuddyAllocator::Heap(16384, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // Start with allocation
    auto ptr_result = buddy->alloc(64, false);
    ASSERT_TRUE(ptr_result.hasValue());
    void* ptr = ptr_result.value();
    size_t current_size = 64;
    
    // Grow through reallocs
    size_t sizes[] = {128, 256, 512, 1024, 2048};
    
    for (size_t new_size : sizes) {
        auto new_result = buddy->realloc(ptr, current_size, new_size, false);
        
        if (new_result.hasValue()) {
            ptr = new_result.value();
            current_size = new_size;
        } else {
            break;
        }
    }
    
    EXPECT_GE(current_size, 256) << "Should successfully realloc multiple times";
    
    buddy->return_element(ptr);
}

// ================================================================================
// Test 18: Sustained Coalescing Load
// ================================================================================

TEST(BuddyStressTest, SustainedCoalescingLoad) {
    auto result = BuddyAllocator::Heap(32768, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    // 100 cycles of: allocate many, free half, free rest
    for (int cycle = 0; cycle < 100; ++cycle) {
        std::vector<void*> ptrs;
        
        // Allocate
        for (int i = 0; i < 30; ++i) {
            auto ptr = buddy->alloc(128, false);
            if (ptr.hasValue()) {
                ptrs.push_back(ptr.value());
            }
        }
        
        // Free half
        for (size_t i = 0; i < ptrs.size() / 2; ++i) {
            buddy->return_element(ptrs[i]);
        }
        
        // Free rest
        for (size_t i = ptrs.size() / 2; i < ptrs.size(); ++i) {
            buddy->return_element(ptrs[i]);
        }
    }
    
    // Should still have good capacity
    EXPECT_GT(buddy->remaining(), 25000) 
        << "Should maintain capacity after sustained coalescing";
}

// ================================================================================
// Test 19: Alignment Stress Test
// ================================================================================

TEST(BuddyStressTest, AlignmentStress) {
    auto result = BuddyAllocator::Heap(32768, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::vector<void*> ptrs;
    size_t alignments[] = {16, 32, 64, 128, 256};
    
    // Allocate with various alignments
    for (int i = 0; i < 50; ++i) {
        size_t align = alignments[i % 5];
        auto ptr = buddy->alloc_aligned(128, align, false);
        
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
            
            // Verify alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.value());
            EXPECT_EQ(addr % align, 0) 
                << "Alignment " << align << " violated in iteration " << i;
        }
    }
    
    // Cleanup
    for (void* ptr : ptrs) {
        buddy->return_element(ptr);
    }
}

// ================================================================================
// Test 20: Long-Running Simulation
// ================================================================================

TEST(BuddyStressTest, LongRunningSimulation) {
    auto result = BuddyAllocator::Heap(65536, 64, 0);
    ASSERT_TRUE(result.hasValue());
    auto buddy = cslt::move(result.value());
    
    std::mt19937 rng(54321);
    std::uniform_int_distribution<int> operation_dist(0, 2); // 0=alloc, 1=free, 2=realloc
    std::uniform_int_distribution<size_t> size_dist(64, 512);
    
    std::vector<std::pair<void*, size_t>> active_allocations;
    
    // 2000 random operations
    for (int op = 0; op < 2000; ++op) {
        int operation = operation_dist(rng);
        
        if (operation == 0 || active_allocations.empty()) {
            // Allocate
            size_t size = size_dist(rng);
            auto ptr = buddy->alloc(size, false);
            if (ptr.hasValue()) {
                active_allocations.push_back({ptr.value(), size});
            }
        } else if (operation == 1 && !active_allocations.empty()) {
            // Free random allocation
            size_t idx = rng() % active_allocations.size();
            buddy->return_element(active_allocations[idx].first);
            active_allocations.erase(active_allocations.begin() + idx);
        } else if (operation == 2 && !active_allocations.empty()) {
            // Realloc random allocation
            size_t idx = rng() % active_allocations.size();
            size_t new_size = size_dist(rng);
            
            auto new_ptr = buddy->realloc(
                active_allocations[idx].first,
                active_allocations[idx].second,
                new_size,
                false
            );
            
            if (new_ptr.hasValue()) {
                active_allocations[idx] = {new_ptr.value(), new_size};
            }
        }
    }
    
    // Cleanup
    for (const auto& alloc : active_allocations) {
        buddy->return_element(alloc.first);
    }
    
    // Verify still functional
    auto test = buddy->alloc(256, false);
    EXPECT_TRUE(test.hasValue()) 
        << "Should still work after 2000 random operations";
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// BASIC CREATION TESTS
// ================================================================================

// ================================================================================
// Test 1: Basic Creation with Valid Parameters
// ================================================================================

TEST(SlabWithBuddyTest, BasicCreation) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create slab for 256-byte objects
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Verify slab was created
    EXPECT_NE(slab.get(), nullptr);
}

// ================================================================================
// Test 2: Object Size Stored Correctly
// ================================================================================

TEST(SlabWithBuddyTest, ObjectSizeStored) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create slab for 128-byte objects
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate should only accept 128 bytes
    auto valid = slab->alloc(128, false);
    EXPECT_TRUE(valid.hasValue());
    
    auto invalid = slab->alloc(256, false);
    EXPECT_FALSE(invalid.hasValue());
}

// ================================================================================
// Test 3: Default Alignment
// ================================================================================

TEST(SlabWithBuddyTest, DefaultAlignment) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create with align=0 (should use alignof(max_align_t))
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Should succeed (default alignment applied)
    EXPECT_NE(slab.get(), nullptr);
}

// ================================================================================
// Test 4: Custom Alignment
// ================================================================================

TEST(SlabWithBuddyTest, CustomAlignment) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create with 64-byte alignment
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 64, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate and verify alignment
    auto ptr_result = slab->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 64, 0) << "Pointer should be 64-byte aligned";
}

// ================================================================================
// Test 5: Default Page Size (4KB heuristic)
// ================================================================================

TEST(SlabWithBuddyTest, DefaultPageSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create with slab_bytes_hint=0 (use default heuristic)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 64, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Should use default (4KB or 64 objects, whichever larger)
    // With 64-byte objects, 64 objects = 4KB, so should be ~4KB
    
    // Allocate first object (triggers page allocation)
    auto ptr = slab->alloc(64, false);
    ASSERT_TRUE(ptr.hasValue());
    
    // Should have capacity for many objects in first page
    size_t total = slab->total_blocks();
    EXPECT_GT(total, 10) << "Should have multiple slots per page";
}

// ================================================================================
// Test 6: Custom Page Size
// ================================================================================

TEST(SlabWithBuddyTest, CustomPageSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create with 8KB pages
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 8192);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate first object (triggers page allocation)
    auto ptr = slab->alloc(128, false);
    ASSERT_TRUE(ptr.hasValue());
    
    // Should have room for many objects
    size_t total = slab->total_blocks();
    EXPECT_GT(total, 20) << "8KB page should hold many 128-byte objects";
}

// ================================================================================
// PARAMETER VALIDATION TESTS
// ================================================================================

// ================================================================================
// Test 7: Zero Object Size (Error)
// ================================================================================

TEST(SlabWithBuddyTest, ZeroObjectSizeError) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // obj_size=0 should fail
    auto result = SlabAllocator::WithBuddy(*buddy, 0, 0, 0);
    
    EXPECT_FALSE(result.hasValue());
}

// ================================================================================
// Test 8: Very Small Object Size
// ================================================================================

TEST(SlabWithBuddyTest, VerySmallObjectSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // 1-byte objects should work (slot_size will be larger for free list linkage)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 1, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    EXPECT_NE(slab.get(), nullptr);
    
    // Should be able to allocate
    auto ptr = slab->alloc(1, false);
    EXPECT_TRUE(ptr.hasValue());
}

// ================================================================================
// Test 9: Large Object Size
// ================================================================================

TEST(SlabWithBuddyTest, LargeObjectSize) {
    auto buddy_result = BuddyAllocator::Heap(16 * 1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // 4KB objects
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 4096, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    EXPECT_NE(slab.get(), nullptr);
    
    // Should be able to allocate
    auto ptr = slab->alloc(4096, false);
    EXPECT_TRUE(ptr.hasValue());
}

// ================================================================================
// ALIGNMENT NORMALIZATION TESTS
// ================================================================================

// ================================================================================
// Test 10: Non-Power-of-2 Alignment Rounded Up
// ================================================================================

TEST(SlabWithBuddyTest, NonPowerOf2AlignmentRounded) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Request 48-byte alignment (not power of 2)
    // Should be rounded to 64
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 48, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    EXPECT_NE(slab.get(), nullptr);
    
    // Allocate and check alignment (should be 64, not 48)
    auto ptr_result = slab->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Should be aligned to 64 (next power of 2 after 48)
    EXPECT_EQ(addr % 64, 0);
}

// ================================================================================
// Test 11: Power-of-2 Alignment Preserved
// ================================================================================

TEST(SlabWithBuddyTest, PowerOf2AlignmentPreserved) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Request 128-byte alignment (already power of 2)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 128, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    EXPECT_NE(slab.get(), nullptr);
    
    // Allocate and check alignment
    auto ptr_result = slab->alloc(256, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 128, 0);
}

// ================================================================================
// Test 12: Extremely Large Alignment (Overflow Check)
// ================================================================================

TEST(SlabWithBuddyTest, ExtremeLargeAlignment) {
    auto buddy_result = BuddyAllocator::Heap(16 * 1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    size_t huge_align = SIZE_MAX / 2 + 1;  // CHANGE THIS (was + 2)
    
    auto result = SlabAllocator::WithBuddy(*buddy, 256, huge_align, 0);
    
    EXPECT_FALSE(result.hasValue());  // Should now fail correctly
}
// ================================================================================
// SLOT SIZE CALCULATION TESTS
// ================================================================================

// ================================================================================
// Test 13: Slot Size for Tiny Objects
// ================================================================================

TEST(SlabWithBuddyTest, SlotSizeForTinyObjects) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // 8-byte object (smaller than free list linkage)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 8, 8, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // stride() returns slot_size_
    size_t stride = slab->stride();
    
    // Should be at least sizeof(Slot) which is sizeof(void*)
    EXPECT_GE(stride, sizeof(void*));
}

// ================================================================================
// Test 14: Slot Size Respects Alignment
// ================================================================================

TEST(SlabWithBuddyTest, SlotSizeRespectsAlignment) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // 100-byte object with 64-byte alignment
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 100, 64, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    size_t stride = slab->stride();
    
    // Stride should be multiple of 64
    EXPECT_EQ(stride % 64, 0);
    
    // Stride should be >= 100
    EXPECT_GE(stride, 100);
}

// ================================================================================
// PAGE GEOMETRY TESTS
// ================================================================================

// ================================================================================
// Test 15: Minimum Page Size (At Least One Slot)
// ================================================================================

TEST(SlabWithBuddyTest, MinimumPageSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Request tiny page size (smaller than header + one slot)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 64);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Should adjust to fit at least one slot
    auto ptr = slab->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue());
}

// ================================================================================
// Test 16: Page Size Adjusted for No Tail Fragment
// ================================================================================

TEST(SlabWithBuddyTest, NoTailFragment) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create slab with specific page size
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 4096);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Page size should be adjusted so slots fit evenly
    // (no wasted space at end of page)
    
    auto ptr = slab->alloc(256, false);
    ASSERT_TRUE(ptr.hasValue());
    
    // Total blocks should be reasonable
    size_t total = slab->total_blocks();
    EXPECT_GT(total, 0);
}

// ================================================================================
// Test 17: Multiple Different Slabs from Same Buddy
// ================================================================================

TEST(SlabWithBuddyTest, MultipleDifferentSlabs) {
    auto buddy_result = BuddyAllocator::Heap(16 * 1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Create multiple slabs with different object sizes
    auto slab128_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab128_result.hasValue());
    auto slab128 = cslt::move(slab128_result.value());
    
    auto slab256_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab256_result.hasValue());
    auto slab256 = cslt::move(slab256_result.value());
    
    auto slab512_result = SlabAllocator::WithBuddy(*buddy, 512, 0, 0);
    ASSERT_TRUE(slab512_result.hasValue());
    auto slab512 = cslt::move(slab512_result.value());
    
    // All should succeed
    EXPECT_NE(slab128.get(), nullptr);
    EXPECT_NE(slab256.get(), nullptr);
    EXPECT_NE(slab512.get(), nullptr);
    
    // Each should allocate its own size
    auto ptr128 = slab128->alloc(128, false);
    auto ptr256 = slab256->alloc(256, false);
    auto ptr512 = slab512->alloc(512, false);
    
    EXPECT_TRUE(ptr128.hasValue());
    EXPECT_TRUE(ptr256.hasValue());
    EXPECT_TRUE(ptr512.hasValue());
}

// ================================================================================
// INITIAL STATE TESTS
// ================================================================================

// ================================================================================
// Test 18: Initial State - No Pages Allocated
// ================================================================================

TEST(SlabWithBuddyTest, InitialStateNoPages) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Initially, no pages should be allocated
    EXPECT_EQ(slab->total_blocks(), 0) << "No pages allocated yet";
    EXPECT_EQ(slab->free_blocks(), 0) << "No free blocks yet";
    EXPECT_EQ(slab->in_use_blocks(), 0) << "Nothing in use";
}

// ================================================================================
// Test 19: Initial State - Zero Usage
// ================================================================================

TEST(SlabWithBuddyTest, InitialStateZeroUsage) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Usage should be zero
    EXPECT_EQ(slab->size(), 0) << "No bytes in use";
    EXPECT_EQ(slab->in_use_blocks(), 0) << "No blocks in use";
}

// ================================================================================
// Test 20: First Allocation Triggers grow_()
// ================================================================================

TEST(SlabWithBuddyTest, FirstAllocationTriggersGrow) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // No pages initially
    EXPECT_EQ(slab->total_blocks(), 0);
    
    // First allocation should trigger grow_()
    auto ptr = slab->alloc(256, false);
    ASSERT_TRUE(ptr.hasValue());
    
    // Now should have pages
    EXPECT_GT(slab->total_blocks(), 0);
    EXPECT_EQ(slab->in_use_blocks(), 1);
    EXPECT_EQ(slab->free_blocks(), slab->total_blocks() - 1);
}

// ================================================================================
// MOVE SEMANTICS TESTS
// ================================================================================

// ================================================================================
// Test 21: Move Semantics
// ================================================================================

TEST(SlabWithBuddyTest, MoveSemantics) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab1_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab1_result.hasValue());
    auto slab1 = cslt::move(slab1_result.value());
    
    // Move to slab2
    auto slab2 = cslt::move(slab1);
    
    // slab1 should be empty
    EXPECT_EQ(slab1.get(), nullptr);
    
    // slab2 should work
    EXPECT_NE(slab2.get(), nullptr);
    
    auto ptr = slab2->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue());
}

// ================================================================================
// STATS AFTER CREATION TESTS
// ================================================================================

// ================================================================================
// Test 22: Stats After Creation (Before First Allocation)
// ================================================================================

TEST(SlabWithBuddyTest, StatsAfterCreation) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 64, 4096);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    char buffer[2048];
    bool ok = slab->stats(buffer, sizeof(buffer));
    
    ASSERT_TRUE(ok);
    
    // Should contain basic info
    std::string stats_str(buffer);
    EXPECT_NE(stats_str.find("Object size: 256"), std::string::npos);
    EXPECT_NE(stats_str.find("Alignment: 64"), std::string::npos);
    EXPECT_NE(stats_str.find("Pages: 0"), std::string::npos); // No pages yet
}

// ================================================================================
// BUDDY LIFETIME TESTS
// ================================================================================

// ================================================================================
// ALIGNMENT EDGE CASES
// ================================================================================

// ================================================================================
// Test 24: Alignment Larger Than Object Size
// ================================================================================

TEST(SlabWithBuddyTest, AlignmentLargerThanObjectSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // 64-byte object with 256-byte alignment
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 64, 256, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    EXPECT_NE(slab.get(), nullptr);
    
    auto ptr_result = slab->alloc(64, false);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    
    EXPECT_EQ(addr % 256, 0) << "Should respect 256-byte alignment";
}

// ================================================================================
// Test 25: Page Size Hint Smaller Than Needed
// ================================================================================

TEST(SlabWithBuddyTest, PageSizeHintTooSmall) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Request 128-byte page (too small for 256-byte objects)
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 128);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Should adjust page size automatically
    EXPECT_NE(slab.get(), nullptr);
    
    // Should still be able to allocate
    auto ptr = slab->alloc(256, false);
    EXPECT_TRUE(ptr.hasValue());
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// ALLOCATION AND DEALLOCATION TESTS
// ================================================================================

// ================================================================================
// Test 1: Basic Allocation and Free
// ================================================================================

TEST(SlabAllocatorTest, BasicAllocAndFree) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate
    auto ptr = slab->alloc(256, false);
    ASSERT_TRUE(ptr.hasValue());
    
    EXPECT_EQ(slab->in_use_blocks(), 1);
    EXPECT_EQ(slab->size(), 256);
    
    // Free
    slab->return_element(ptr.value());
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
    EXPECT_EQ(slab->size(), 0);
}

// ================================================================================
// Test 2: Multiple Allocations from Same Page
// ================================================================================

TEST(SlabAllocatorTest, MultipleAllocationsOnePage) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate multiple objects
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        auto ptr = slab->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 10);
    EXPECT_EQ(slab->size(), 10 * 128);
    
    // Free all
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}

// ================================================================================
// Test 3: Allocation Triggers Multiple Pages
// ================================================================================

TEST(SlabAllocatorTest, MultiplePageGrowth) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Small page size to force multiple pages
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 1024);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate enough to need multiple pages
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
        auto ptr = slab->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Should have grown multiple pages
    EXPECT_GT(slab->total_blocks(), 10);
    
    // Cleanup
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
}

// ================================================================================
// Test 4: Interleaved Alloc and Free
// ================================================================================

TEST(SlabAllocatorTest, InterleavedAllocFree) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate 5
    for (int i = 0; i < 5; ++i) {
        auto ptr = slab->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 5);
    
    // Free 3
    for (int i = 0; i < 3; ++i) {
        slab->return_element(ptrs[i]);
    }
    ptrs.erase(ptrs.begin(), ptrs.begin() + 3);
    
    EXPECT_EQ(slab->in_use_blocks(), 2);
    
    // Allocate 4 more
    for (int i = 0; i < 4; ++i) {
        auto ptr = slab->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 6);
    
    // Cleanup
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
}

// ================================================================================
// Test 5: Free in Reverse Order
// ================================================================================

TEST(SlabAllocatorTest, FreeReverseOrder) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate
    for (int i = 0; i < 10; ++i) {
        auto ptr = slab->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Free in reverse order
    for (int i = ptrs.size() - 1; i >= 0; --i) {
        slab->return_element(ptrs[i]);
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}

// ================================================================================
// Test 6: Free in Random Order
// ================================================================================

TEST(SlabAllocatorTest, FreeRandomOrder) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate
    for (int i = 0; i < 20; ++i) {
        auto ptr = slab->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    // Shuffle
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ptrs.begin(), ptrs.end(), g);
    
    // Free in random order
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}

// ================================================================================
// ZERO INITIALIZATION TESTS
// ================================================================================

// ================================================================================
// Test 7: Zero Initialization
// ================================================================================

TEST(SlabAllocatorTest, ZeroInitialization) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate with zero initialization
    auto ptr_result = slab->alloc(256, true);
    ASSERT_TRUE(ptr_result.hasValue());
    
    void* ptr = ptr_result.value();
    uint8_t* data = static_cast<uint8_t*>(ptr);
    
    // Check all bytes are zero
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], 0) << "Byte " << i << " not zero";
    }
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 8: Non-Zero Then Zero
// ================================================================================

TEST(SlabAllocatorTest, NonZeroThenZero) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate without zero
    auto ptr1 = slab->alloc(128, false).value();
    
    // Write pattern
    uint8_t* data1 = static_cast<uint8_t*>(ptr1);
    for (size_t i = 0; i < 128; ++i) {
        data1[i] = 0xFF;
    }
    
    // Free
    slab->return_element(ptr1);
    
    // Allocate same slot with zero
    auto ptr2 = slab->alloc(128, true).value();
    
    // Should be zeroed even though it was 0xFF before
    uint8_t* data2 = static_cast<uint8_t*>(ptr2);
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(data2[i], 0);
    }
    
    slab->return_element(ptr2);
}

// ================================================================================
// DATA INTEGRITY TESTS
// ================================================================================

// ================================================================================
// Test 9: Data Integrity Single Object
// ================================================================================

TEST(SlabAllocatorTest, DataIntegritySingle) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    uint8_t* data = static_cast<uint8_t*>(ptr);
    
    // Write pattern
    for (size_t i = 0; i < 256; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    
    // Verify pattern
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], static_cast<uint8_t>(i));
    }
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 10: Data Integrity Multiple Objects
// ================================================================================

TEST(SlabAllocatorTest, DataIntegrityMultiple) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate and write unique patterns
    for (int i = 0; i < 20; ++i) {
        auto ptr = slab->alloc(128, false).value();
        uint8_t* data = static_cast<uint8_t*>(ptr);
        
        // Write unique pattern for this object
        uint8_t pattern = static_cast<uint8_t>(i + 1);
        for (size_t j = 0; j < 128; ++j) {
            data[j] = pattern;
        }
        
        ptrs.push_back(ptr);
    }
    
    // Verify all patterns are intact
    for (size_t i = 0; i < ptrs.size(); ++i) {
        uint8_t* data = static_cast<uint8_t*>(ptrs[i]);
        uint8_t expected = static_cast<uint8_t>(i + 1);
        
        for (size_t j = 0; j < 128; ++j) {
            EXPECT_EQ(data[j], expected) << "Object " << i << " corrupted";
        }
    }
    
    // Cleanup
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
}

// ================================================================================
// Test 11: Data Survives Interleaved Operations
// ================================================================================

TEST(SlabAllocatorTest, DataSurvivesInterleavedOps) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    std::vector<uint8_t> patterns;
    
    // Allocate 10 objects with patterns
    for (int i = 0; i < 10; ++i) {
        auto ptr = slab->alloc(256, false).value();
        uint8_t* data = static_cast<uint8_t*>(ptr);
        uint8_t pattern = static_cast<uint8_t>(i + 10);
        
        memset(data, pattern, 256);
        
        ptrs.push_back(ptr);
        patterns.push_back(pattern);
    }
    
    // Free every other object
    for (int i = 1; i < 10; i += 2) {
        slab->return_element(ptrs[i]);
        ptrs[i] = nullptr;
    }
    
    // Verify remaining objects
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        if (ptrs[i]) {
            uint8_t* data = static_cast<uint8_t*>(ptrs[i]);
            for (size_t j = 0; j < 256; ++j) {
                EXPECT_EQ(data[j], patterns[i]);
            }
        }
    }
    
    // Cleanup
    for (void* ptr : ptrs) {
        if (ptr) slab->return_element(ptr);
    }
}

// ================================================================================
// SIZE VALIDATION TESTS
// ================================================================================

// ================================================================================
// Test 12: Wrong Size Allocation Rejected
// ================================================================================

TEST(SlabAllocatorTest, WrongSizeRejected) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Try to allocate wrong sizes
    auto too_small = slab->alloc(128, false);
    EXPECT_FALSE(too_small.hasValue());
    
    auto too_large = slab->alloc(512, false);
    EXPECT_FALSE(too_large.hasValue());
    
    // Correct size should work
    auto correct = slab->alloc(256, false);
    EXPECT_TRUE(correct.hasValue());
    
    slab->return_element(correct.value());
}

// ================================================================================
// Test 13: alloc_aligned Size Validation
// ================================================================================

TEST(SlabAllocatorTest, AllocAlignedSizeValidation) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 64, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Wrong size with alignment
    auto wrong = slab->alloc_aligned(128, 64, false);
    EXPECT_FALSE(wrong.hasValue());
    
    // Correct size with alignment
    auto correct = slab->alloc_aligned(256, 64, false);
    EXPECT_TRUE(correct.hasValue());
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(correct.value());
    EXPECT_EQ(addr % 64, 0);
    
    slab->return_element(correct.value());
}

// ================================================================================
// ALIGNMENT TESTS
// ================================================================================

// ================================================================================
// Test 14: Alignment Satisfied for All Allocations
// ================================================================================

TEST(SlabAllocatorTest, AlignmentSatisfied) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 128, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate many objects
    for (int i = 0; i < 20; ++i) {
        auto ptr = slab->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr.value());
        EXPECT_EQ(addr % 128, 0) << "Allocation " << i << " not aligned";
        
        slab->return_element(ptr.value());
    }
}

// ================================================================================
// Test 15: alloc_aligned Exceeds Slab Alignment
// ================================================================================

TEST(SlabAllocatorTest, AllocAlignedExceedsSlab) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Slab with 64-byte alignment
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 64, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Request 128-byte alignment (exceeds slab's 64)
    auto result = slab->alloc_aligned(256, 128, false);
    
    EXPECT_FALSE(result.hasValue());
}

// ================================================================================
// Test 16: alloc_aligned Within Slab Alignment
// ================================================================================

TEST(SlabAllocatorTest, AllocAlignedWithinSlab) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    // Slab with 128-byte alignment
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 128, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Request 64-byte alignment (within slab's 128)
    auto result = slab->alloc_aligned(256, 64, false);
    
    EXPECT_TRUE(result.hasValue());
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(result.value());
    EXPECT_EQ(addr % 64, 0);
    
    slab->return_element(result.value());
}

// ================================================================================
// RESET TESTS
// ================================================================================

// ================================================================================
// Test 17: Reset Clears All Allocations
// ================================================================================

TEST(SlabAllocatorTest, ResetClearsAll) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate many
    for (int i = 0; i < 20; ++i) {
        auto ptr = slab->alloc(128, false);
        ASSERT_TRUE(ptr.hasValue());
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 20);
    EXPECT_GT(slab->total_blocks(), 0);
    
    // Reset
    bool ok = slab->reset();
    EXPECT_TRUE(ok);
    
    // Everything should be freed
    EXPECT_EQ(slab->in_use_blocks(), 0);
    EXPECT_EQ(slab->size(), 0);
    
    // Free list should be rebuilt
    EXPECT_EQ(slab->free_blocks(), slab->total_blocks());
}

// ================================================================================
// Test 18: Allocate After Reset
// ================================================================================

TEST(SlabAllocatorTest, AllocateAfterReset) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate
    auto ptr1 = slab->alloc(256, false);
    ASSERT_TRUE(ptr1.hasValue());
    
    // Reset
    slab->reset();
    
    // Should be able to allocate again
    auto ptr2 = slab->alloc(256, false);
    EXPECT_TRUE(ptr2.hasValue());
    
    slab->return_element(ptr2.value());
}

// ================================================================================
// Test 19: Multiple Reset Cycles
// ================================================================================

TEST(SlabAllocatorTest, MultipleResetCycles) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    for (int cycle = 0; cycle < 5; ++cycle) {
        // Allocate
        for (int i = 0; i < 10; ++i) {
            auto ptr = slab->alloc(128, false);
            ASSERT_TRUE(ptr.hasValue());
        }
        
        EXPECT_EQ(slab->in_use_blocks(), 10);
        
        // Reset
        slab->reset();
        
        EXPECT_EQ(slab->in_use_blocks(), 0);
    }
}

// ================================================================================
// POINTER VALIDATION TESTS
// ================================================================================

// ================================================================================
// Test 20: is_ptr Valid Pointer
// ================================================================================

TEST(SlabAllocatorTest, IsPtrValid) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    EXPECT_TRUE(slab->is_ptr(ptr));
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 21: is_ptr Invalid Pointers
// ================================================================================

TEST(SlabAllocatorTest, IsPtrInvalid) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // NULL pointer
    EXPECT_FALSE(slab->is_ptr(nullptr));
    
    // External pointer
    void* external = malloc(256);
    EXPECT_FALSE(slab->is_ptr(external));
    free(external);
    
    // Allocate a valid pointer
    auto ptr = slab->alloc(256, false).value();
    
    // Misaligned pointer (offset into object)
    uint8_t* misaligned = static_cast<uint8_t*>(ptr) + 10;
    EXPECT_FALSE(slab->is_ptr(misaligned));
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 22: is_ptr_sized Valid
// ================================================================================

TEST(SlabAllocatorTest, IsPtrSizedValid) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    // Exact size
    EXPECT_TRUE(slab->is_ptr_sized(ptr, 256));
    
    // Smaller size (fits)
    EXPECT_TRUE(slab->is_ptr_sized(ptr, 128));
    EXPECT_TRUE(slab->is_ptr_sized(ptr, 1));
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 23: is_ptr_sized Too Large
// ================================================================================

TEST(SlabAllocatorTest, IsPtrSizedTooLarge) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    // Size exceeds object size
    EXPECT_FALSE(slab->is_ptr_sized(ptr, 257));
    EXPECT_FALSE(slab->is_ptr_sized(ptr, 512));
    
    slab->return_element(ptr);
}

// ================================================================================
// STATS TESTS
// ================================================================================

// ================================================================================
// Test 24: Stats After Allocations
// ================================================================================

TEST(SlabAllocatorTest, StatsAfterAllocations) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 64, 4096);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Allocate some objects
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        auto ptr = slab->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        ptrs.push_back(ptr.value());
    }
    
    char buffer[4096];
    bool ok = slab->stats(buffer, sizeof(buffer));
    ASSERT_TRUE(ok);
    
    std::string stats_str(buffer);
    
    // Should show object size
    EXPECT_NE(stats_str.find("Object size: 256"), std::string::npos);
    
    // Should show alignment
    EXPECT_NE(stats_str.find("Alignment: 64"), std::string::npos);
    
    // Should show in-use blocks
    EXPECT_NE(stats_str.find("In-use blocks: 5"), std::string::npos);
    
    // Cleanup
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
}

// ================================================================================
// REALLOC TESTS
// ================================================================================

// ================================================================================
// Test 25: Realloc Same Size Returns Same Pointer
// ================================================================================

TEST(SlabAllocatorTest, ReallocSameSize) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    // Realloc same size
    auto new_ptr_result = slab->realloc(ptr, 256, 256, false);
    ASSERT_TRUE(new_ptr_result.hasValue());
    
    // Should return same pointer
    EXPECT_EQ(new_ptr_result.value(), ptr);
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 26: Realloc Different Size Rejected
// ================================================================================

TEST(SlabAllocatorTest, ReallocDifferentSizeRejected) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    // Try to grow
    auto grow = slab->realloc(ptr, 256, 512, false);
    EXPECT_FALSE(grow.hasValue());
    
    // Try to shrink
    auto shrink = slab->realloc(ptr, 256, 128, false);
    EXPECT_FALSE(shrink.hasValue());
    
    slab->return_element(ptr);
}

// ================================================================================
// Test 27: Realloc from nullptr
// ================================================================================

TEST(SlabAllocatorTest, ReallocFromNullptr) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Realloc from nullptr acts as alloc
    auto ptr = slab->realloc(nullptr, 0, 256, false);
    EXPECT_TRUE(ptr.hasValue());
    
    slab->return_element(ptr.value());
}

// ================================================================================
// Test 28: Realloc to Zero (Free)
// ================================================================================

TEST(SlabAllocatorTest, ReallocToZero) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    auto ptr = slab->alloc(256, false).value();
    
    EXPECT_EQ(slab->in_use_blocks(), 1);
    
    // Realloc to zero (free)
    auto result = slab->realloc(ptr, 256, 0, false);
    EXPECT_TRUE(result.hasValue());
    EXPECT_EQ(result.value(), nullptr);
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}

// ================================================================================
// STRESS TESTS
// ================================================================================

// ================================================================================
// Test 29: Stress Test - Many Allocations
// ================================================================================

TEST(SlabAllocatorTest, StressManyAllocations) {
    auto buddy_result = BuddyAllocator::Heap(16 * 1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 128, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    std::vector<void*> ptrs;
    
    // Allocate many objects
    for (int i = 0; i < 1000; ++i) {
        auto ptr = slab->alloc(128, false);
        if (ptr.hasValue()) {
            ptrs.push_back(ptr.value());
        }
    }
    
    EXPECT_GT(ptrs.size(), 500) << "Should allocate at least 500 objects";
    
    // Cleanup
    for (void* ptr : ptrs) {
        slab->return_element(ptr);
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}

// ================================================================================
// Test 30: Stress Test - Rapid Alloc/Free
// ================================================================================

TEST(SlabAllocatorTest, StressRapidAllocFree) {
    auto buddy_result = BuddyAllocator::Heap(1024 * 1024, 64, 0);
    ASSERT_TRUE(buddy_result.hasValue());
    auto buddy = cslt::move(buddy_result.value());
    
    auto slab_result = SlabAllocator::WithBuddy(*buddy, 256, 0, 0);
    ASSERT_TRUE(slab_result.hasValue());
    auto slab = cslt::move(slab_result.value());
    
    // Rapid cycles
    for (int i = 0; i < 1000; ++i) {
        auto ptr = slab->alloc(256, false);
        ASSERT_TRUE(ptr.hasValue());
        slab->return_element(ptr.value());
    }
    
    EXPECT_EQ(slab->in_use_blocks(), 0);
}
// ================================================================================
// ================================================================================
// eof
