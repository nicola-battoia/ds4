#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <inttypes.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "metal-memory-probe: no default Metal device\n");
            return 1;
        }

        printf("device=%s\n", device.name.UTF8String);
        printf("unified_memory=%s\n", device.hasUnifiedMemory ? "yes" : "no");
        printf("recommended_working_set_bytes=%" PRIu64 "\n",
               (uint64_t)device.recommendedMaxWorkingSetSize);
        printf("recommended_working_set_gib=%.3f\n",
               (double)device.recommendedMaxWorkingSetSize /
                   (1024.0 * 1024.0 * 1024.0));
        printf("max_buffer_bytes=%" PRIu64 "\n",
               (uint64_t)device.maxBufferLength);
        printf("current_allocated_bytes=%" PRIu64 "\n",
               (uint64_t)device.currentAllocatedSize);
    }
    return 0;
}
