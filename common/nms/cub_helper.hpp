//
// Created by Administrator on 9/14/2022.
//
#include "kernel.h"
template <typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments) {
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending((void*)NULL, temp_storage_bytes,
                                                       (const KeyT*)NULL, (KeyT*)NULL,
                                                       (const ValueT*)NULL, (ValueT*)NULL,
                                                       num_items,     // # items
                                                       num_segments,  // # segments
                                                       (const int*)NULL, (const int*)NULL);
    return temp_storage_bytes;
}
