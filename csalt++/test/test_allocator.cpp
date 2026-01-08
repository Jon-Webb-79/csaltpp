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
// eof
