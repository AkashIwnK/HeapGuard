/* Header for the HeapGuard */

#ifndef _HEAPGUARD_H_
#define _HEAPGUARD_H_

#include <stdbool.h>

#define HEAPGUARD_INITIALIZED 		(int8_t)(~0)
#define HEAPGUARD_UNINITIALIZED 	(int8_t)(0x1f)

/* These are the primary parameters that can control other paramters.
 * More is the value of this parameter more time it takes for
 * HeapGuard to intialize.
 */
#define HEAP_BAG_SHIFT 31
#define LARGE_HEAP_BAG_SHIFT 33

/* We use a fixed page size for performance purposes */
#define DEFAULT_PAGE_SIZE ((uint64_t)1 << 12)
#define DEFAULT_PAGE_MASK ((uint64_t)(~0) << 12)
#define DEFAULT_PAGE_SHIFT 12

#define SUBHEAP_SHIFT 20
#define LARGE_SUBHEAP_SHIFT 28
#define NUM_SUBHEAPS_SHIFT 4
#define NUM_LARGE_SUBHEAPS_SHIFT 3

#define NUM_SUBHEAPS 		((uint64_t)1 << NUM_SUBHEAPS_SHIFT)
#define NUM_LARGE_SUBHEAPS 	((uint64_t)1 << NUM_LARGE_SUBHEAPS_SHIFT)
#define HEAP_BAG_SIZE  		((uint64_t)1 << HEAP_BAG_SHIFT)
#define LARGE_HEAP_BAG_SIZE ((uint64_t)1 << LARGE_HEAP_BAG_SHIFT)
#define SUBHEAP_SIZE   		((uint64_t)1 << SUBHEAP_SHIFT)
#define LARGE_SUBHEAP_SIZE  ((uint64_t)1 << LARGE_SUBHEAP_SHIFT)
#define HEAP_SIZE      		((uint64_t)1 << (NUM_SUBHEAPS_SHIFT + SUBHEAP_SHIFT))
#define LARGE_HEAP_SIZE     ((uint64_t)1 << (NUM_LARGE_SUBHEAPS_SHIFT + LARGE_SUBHEAP_SHIFT))

#define HEAP_BAG_ALIGN_MASK		 ((uint64_t)(~0) << HEAP_BAG_SHIFT)
#define HEAP_BAG_SIZE_MASK       (~HEAP_BAG_ALIGN_MASK)
#define SUBHEAP_ALIGN_MASK 		 ((uint64_t)(~0) << SUBHEAP_SHIFT)
#define LARGE_SUBHEAP_ALIGN_MASK ((uint64_t)(~0) << LARGE_SUBHEAP_SHIFT)
#define HEAP_ALIGN_MASK			 ((uint64_t)(~0) << (NUM_SUBHEAPS_SHIFT + SUBHEAP_SHIFT))
#define LARGE_HEAP_ALIGN_MASK	 ((uint64_t)(~0) << (NUM_LARGE_SUBHEAPS_SHIFT + LARGE_SUBHEAP_SHIFT))

#define HEAP_BAG_MASK       ((uint64_t)(~0) << HEAP_BAG_SHIFT)
#define LARGE_HEAP_BAG_MASK ((uint64_t)(~0) << LARGE_HEAP_BAG_SHIFT)
#define SUBHEAP_MASK 		((uint64_t)(~0) << SUBHEAP_SHIFT)
#define LARGE_SUBHEAP_MASK  ((uint64_t)(~0) << LARGE_SUBHEAP_SHIFT)
#define HEAP_MASK			((uint64_t)(~0) << (NUM_SUBHEAPS_SHIFT + SUBHEAP_SHIFT))
#define LARGE_HEAP_MASK	 	((uint64_t)(~0) << (NUM_LARGE_SUBHEAPS_SHIFT + LARGE_SUBHEAP_SHIFT))

#define NUM_HEAPS 			(HEAP_BAG_SIZE/HEAP_SIZE)
#define NUM_LARGE_HEAPS		(LARGE_HEAP_BAG_SIZE/LARGE_HEAP_SIZE)

#define MIN_OBJ_SHIFT 		4
#define MIN_LARGE_OBJ_SHIFT 20

#define MIN_OBJ_SIZE 		((uint64_t)1 << MIN_OBJ_SHIFT)
#define MIN_LARGE_OBJ_SIZE 	((uint64_t)1 << MIN_LARGE_OBJ_SHIFT)
#define MAX_SMALL_OBJ_SIZE  (MIN_LARGE_OBJ_SIZE >> 1)

#define LARGEST_PREALLOCATED_SIZE ((uint64_t)1 << 27)

#define INVALID_HEAPOBJ_SIZE  (uint64_t)(~0)

#define HASH_BAG_SHIFT 			MIN_OBJ_SHIFT
#define HASH_LARGE_BAG_SHIFT	MIN_LARGE_OBJ_SHIFT

/* Heap Bag types */
#define SMALL_HEAP_BAG 	1
#define LARGE_HEAP_BAG 	2
#define BIG_BLOCK       3
#define PRIMARY BAG     4

/* Metadata comprises of pointer to free list and the pointer to the adjacent
 * unused heap object.
 */
#define METADATA_SIZE 16

#define SHADOW_MAP_SCALE		(SUBHEAP_SHIFT - 4)
#define LARGE_SHADOW_MAP_SCALE  (LARGE_SUBHEAP_SHIFT - 4)

#define TOTAL_METADATA_SIZE 		(NUM_SUBHEAPS * METADATA_SIZE)
#define TOTAL_LARGE_METADATA_SIZE 	(NUM_LARGE_SUBHEAPS * METADATA_SIZE)

#define SHADOW_METADATA_SIZE (TOTAL_METADATA_SIZE + TOTAL_LARGE_METADATA_SIZE)

#define BAG_SHADOW_SIZE      	HEAP_BAG_SIZE

#define MAX_NUM_THREADS			((uint64_t)1 << 15)
#define THREAD_INFO_NUM_ELEM 	((uint64_t)1 << 28) /* This should be good enough */
#define THREAD_ID_MASK			0x00000fffffff0000

#define BAG_TYPE_HASH_TABLE_SIZE ((uint64_t)1 << 47) >> ((uint64_t)HEAP_BAG_SHIFT)
#define HASH_TABLE_SHIFT        HEAP_BAG_SHIFT

/* 500 KB of space for bag info space should be good enough */
#define BAG_INFO_SPACE_SIZE     ((uint64_t)1 << 19)

#define BAG_INFO_ENCODING_SHIFT 	48
#define BAG_INFO_ENCODING_MASK     (uint64_t)(~((uint64_t)(~0) << BAG_INFO_ENCODING_SHIFT))

#define SECR_COMP_MASK    	   ((uint64_t)(~0) << 48)
#define SECR_COMP_SHIFT        48
#define SMALL_ADDR_DIFF_MASK  ~((uint64_t)(~0) << 25) & ((uint64_t)(~0) << 9)
#define LARGE_ADDR_DIFF_MASK   ((uint64_t)(~0) << 20) & ~((uint64_t)(~0) << 36) /* Set 16 or less bits */


/************** INITIALIZER *************/
void Init_HeapGuard(void) __attribute__((constructor));

/************* HEAP MANAGER API **************/
void *Heap_malloc(uint64_t objSize);
void Heap_free(void *heapObjPtr);
void *Heap_calloc(uint64_t numObj, uint64_t objSize);
void *Heap_realloc(void *heapObjPtr, uint64_t objSize);

/************* FUNTION WRAPPERS ************/
void Redefine_Heap_Mmap(void *(*new_mmap)(void *, size_t, int, int, int, off_t));
void Redefine_Heap_Shadow_Mmap(void *(*new_mmap)(void *, size_t, int, int, int, off_t));
void Redefine_Heap_Munmap(int (*new_munmap)(void *, size_t));
void Redefine_Heap_Shadow_Munmap(int (*new_munmap)(void *, size_t));
void Redefine_Heap_Mprotect(int (*new_mprotect)(void *, size_t, int));
void Redefine_Heap_Shadow_Mprotect(int (*new_mprotect)(void *, size_t, int));
void Redefine_Memcpy(void *(*new_memcpy)(void *, const void *, size_t));
void Redefine_Memset(void *(*new_memset)(void *, int, size_t));

/*************** DEBUGING FUNCTIONS *************/
void TrackLoads(uint64_t loadAddr);
void TrackStores(uint64_t storeAddr);

#endif /* _HEAPGUARD_H_ */
