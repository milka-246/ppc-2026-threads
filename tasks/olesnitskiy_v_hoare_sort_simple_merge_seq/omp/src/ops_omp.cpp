#include "olesnitskiy_v_hoare_sort_simple_merge_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "olesnitskiy_v_hoare_sort_simple_merge_seq/common/include/common.hpp"

namespace olesnitskiy_v_hoare_sort_simple_merge_seq {

OlesnitskiyVHoareSortSimpleMergeOMP::OlesnitskiyVHoareSortSimpleMergeOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

int OlesnitskiyVHoareSortSimpleMergeOMP::HoarePartition(std::vector<int> &array, int left, int right) {
  const int pivot = array[left + ((right - left) / 2)];
  int i = left - 1;
  int j = right + 1;

  while (true) {
    ++i;
    while (array[i] < pivot) {
      ++i;
    }

    --j;
    while (array[j] > pivot) {
      --j;
    }

    if (i >= j) {
      return j;
    }

    std::swap(array[i], array[j]);
  }
}

void OlesnitskiyVHoareSortSimpleMergeOMP::HoareQuickSort(std::vector<int> &array, int left, int right) {
  std::stack<std::pair<int, int>> ranges;
  ranges.emplace(left, right);

  while (!ranges.empty()) {
    auto [current_left, current_right] = ranges.top();
    ranges.pop();

    if (current_left >= current_right) {
      continue;
    }

    const int partition_index = HoarePartition(array, current_left, current_right);

    if ((partition_index - current_left) > (current_right - (partition_index + 1))) {
      ranges.emplace(current_left, partition_index);
      ranges.emplace(partition_index + 1, current_right);
    } else {
      ranges.emplace(partition_index + 1, current_right);
      ranges.emplace(current_left, partition_index);
    }
  }
}

void OlesnitskiyVHoareSortSimpleMergeOMP::Merge(std::vector<int> &array, int left, int mid, int right) {
  std::vector<int> merged;
  const int merged_size = (right - left) + 1;
  merged.reserve(static_cast<std::size_t>(merged_size));

  int left_index = left;
  int right_index = mid + 1;

  while (left_index <= mid && right_index <= right) {
    if (array[left_index] <= array[right_index]) {
      merged.push_back(array[left_index++]);
    } else {
      merged.push_back(array[right_index++]);
    }
  }

  while (left_index <= mid) {
    merged.push_back(array[left_index++]);
  }

  while (right_index <= right) {
    merged.push_back(array[right_index++]);
  }

  for (std::size_t index = 0; index < merged.size(); ++index) {
    array[static_cast<std::size_t>(left) + index] = merged[index];
  }
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::RunImpl() {
  std::vector<int> &array = GetOutput();
  const int size = static_cast<int>(array.size());
  if (size <= 1) {
    return true;
  }

  const int max_threads = std::max(1, omp_get_max_threads());
  const int chunk_count = std::min(max_threads, size);

  if (chunk_count == 1) {
    HoareQuickSort(array, 0, size - 1);
    return std::ranges::is_sorted(array);
  }

  std::vector<int> borders(static_cast<std::size_t>(chunk_count + 1));
  for (int border_index = 0; border_index <= chunk_count; ++border_index) {
    borders[static_cast<std::size_t>(border_index)] = (border_index * size) / chunk_count;
  }

#pragma omp parallel for default(none) shared(array, borders, chunk_count)
  for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const int left = borders[static_cast<std::size_t>(chunk_index)];
    const int right = borders[static_cast<std::size_t>(chunk_index) + 1] - 1;
    if (left < right) {
      HoareQuickSort(array, left, right);
    }
  }

  for (int merge_index = 0; merge_index < chunk_count - 1; ++merge_index) {
    const int mid = borders[static_cast<std::size_t>(merge_index) + 1] - 1;
    const int right = borders[static_cast<std::size_t>(merge_index) + 2] - 1;
    Merge(array, 0, mid, right);
  }

  return std::ranges::is_sorted(array);
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::PostProcessingImpl() {
  return !GetOutput().empty() && std::ranges::is_sorted(GetOutput());
}

}  // namespace olesnitskiy_v_hoare_sort_simple_merge_seq
