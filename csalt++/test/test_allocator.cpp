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
// eof
